// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.ml
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.functions.{col, explode, udf}
import org.apache.spark.sql.{Dataset, Row}

/**
  * ML version of IndexedRow
  *
  * @param index  Index of row
  * @param vector Sparse or Dense form of the vector for the provided index
  */
case class IndexedRow(index: Long, vector: Vector)

/**
  * Dense Matrix wrapped for Spark Dataset of IndexRow
  *
  * Based on the requirements for Spark's IndexedRowMatrix
  *
  * @param rows  list of the index, vector row pairs
  * @param nRows number of rows, in case padding is needed
  * @param nCols number of columns, in case padding is needed
  */
class IndexedRowMatrix(val rows: Dataset[IndexedRow],
                       private var nRows: Long,
                       private var nCols: Long) extends DistributedMatrix {

  /**
    * Converts to BlockMatrix. Creates blocks with size 1024 x 1024.
    */
  def toBlockMatrix(): BlockMatrix = {
    toBlockMatrix(1024, 1024)
  }

  /**
    * Converts to BlockMatrix. Blocks may be sparse or dense depending on the sparsity of the rows.
    *
    * @param rowsPerBlock The number of rows of each block. The blocks at the bottom edge may have
    *                     a smaller value. Must be an integer value greater than 0.
    * @param colsPerBlock The number of columns of each block. The blocks at the right edge may have
    *                     a smaller value. Must be an integer value greater than 0.
    * @return a [[BlockMatrix]]
    */
  def toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")

    val m = numRows()
    val n = numCols()

    // Since block matrices require an integer row index
    require(math.ceil(m.toDouble / rowsPerBlock) <= Int.MaxValue,
      "Number of rows divided by rowsPerBlock cannot exceed maximum integer.")

    // The remainder calculations only matter when m % rowsPerBlock != 0 or n % colsPerBlock != 0
    val remainderRowBlockIndex = m / rowsPerBlock
    val remainderColBlockIndex = n / colsPerBlock
    val remainderRowBlockSize = (m % rowsPerBlock).toInt
    val remainderColBlockSize = (n % colsPerBlock).toInt
    val numRowBlocks = math.ceil(m.toDouble / rowsPerBlock).toInt
    val numColBlocks = math.ceil(n.toDouble / colsPerBlock).toInt

    val blocks = rows.rdd.flatMap { ir: IndexedRow =>
      val blockRow = ir.index / rowsPerBlock
      val rowInBlock = ir.index % rowsPerBlock

      ir.vector match {
        case SparseVector(size, indices, values) =>
          indices.zip(values).map { case (index, value) =>
            val blockColumn = index / colsPerBlock
            val columnInBlock = index % colsPerBlock
            ((blockRow.toInt, blockColumn.toInt), (rowInBlock.toInt, Array((value, columnInBlock))))
          }
        case DenseVector(values) =>
          values.grouped(colsPerBlock)
            .zipWithIndex
            .map { case (values, blockColumn) =>
              ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values.zipWithIndex))
            }
      }
    }.groupByKey(GridPartitioner(numRowBlocks, numColBlocks, rows.rdd.getNumPartitions)).map {
      case ((blockRow, blockColumn), itr) =>
        val actualNumRows =
          if (blockRow == remainderRowBlockIndex) remainderRowBlockSize else rowsPerBlock
        val actualNumColumns =
          if (blockColumn == remainderColBlockIndex) remainderColBlockSize else colsPerBlock

        val arraySize = actualNumRows * actualNumColumns
        val matrixAsArray = new Array[Double](arraySize)
        var countForValues = 0
        itr.foreach { case (rowWithinBlock, valuesWithColumns) =>
          valuesWithColumns.foreach { case (value, columnWithinBlock) =>
            matrixAsArray.update(columnWithinBlock * actualNumRows + rowWithinBlock, value)
            countForValues += 1
          }
        }
        val denseMatrix = new DenseMatrix(actualNumRows, actualNumColumns, matrixAsArray)
        val finalMatrix = if (countForValues / arraySize.toDouble >= 0.1) {
          denseMatrix
        } else {
          denseMatrix.toSparse
        }

        ((blockRow, blockColumn), finalMatrix)
    }
    import rows.sparkSession.implicits._
    new BlockMatrix(blocks.toDS(), rowsPerBlock, colsPerBlock, m, n)
  }

  private[ml] def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    rows.collect().foreach { case IndexedRow(rowIndex, vector) =>
      val i = rowIndex.toInt
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
    }
    mat
  }

  def toRowMatrix(): RowMatrix = {
    //    import rows.sparkSession.implicits._
    //
    //    val toRow = udf((items: ml.linalg.Vector) => {
    //      items
    //    })

    val frame = rows.select(col(rows.columns(1))).asInstanceOf[Dataset[RowVector]]
    new RowMatrix(frame) //.map(row => row.vector)
    //        .select(col(entries.columns(1)))
    //        .map(row => row.getAs[Row](0).getAs[Vector](0)))
  }

  def multiply(other: IndexedRowMatrix): DistributedMatrix = {
    multiply(other.toCoordinateMatrix)
  }

  def multiply(other: CoordinateMatrix): DistributedMatrix = {
    toCoordinateMatrix.multiply(other)
  }

  def toCoordinateMatrix(): CoordinateMatrix = {
    import rows.sparkSession.implicits._

    val zip = udf((items: ml.linalg.Vector) => {
      items.toArray.zipWithIndex.filter(_._1 > 0)
    })

    new CoordinateMatrix(
      rows.select(col(rows.columns(0)), explode(zip(col(rows.columns(1)))))
        .map(row => {
          MatrixEntry(
            row.getLong(0),
            row.getAs[Row](1).getInt(1).toLong,
            row.getAs[Row](1).getDouble(0))
        })
    )
  }

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(entries: Dataset[IndexedRow]) = this(entries, 0L, 0)

  override def numCols(): Long = {
    if (nCols <= 0) {
      // Calling `first` will throw an exception if `rows` is empty.
      nCols = rows.first().vector.size.toLong
    }
    nCols
  }

  override def numRows(): Long = {
    if (nRows <= 0L) {
      nRows = rows.groupBy().max("index").collect()(0).getLong(0) + 1
    }
    nRows
  }
}
