/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.annotation.Since
import org.apache.spark.ml
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vectors => V}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix => CM}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, collect_list, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * Sparse Matrix wrapped for Spark Dataset of MatrixEntry
  *
  * Based on the requirements for Spark's CoordinateMatrix
  * An RDD that would be used to create a CoordinateMatrix, can be used to create a
  * Sparse Matrix using rdd.toDS when it is an RDD of MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val rddMatrixEntry: RDD[MatrixEntry] = rdd.map(row => new MatrixEntry(row._1, row._2, row._3)
  * val sparseMatrix: SparseMatrix = rddMatrixEntry.toDS
  *
  * Or, a DataFrame of Row, can be mapped to a MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val sparseMatrix: SparseMatrix =
  *     dataframe.map(row => new MatrixEntry(row.getLong(0). row.getLong(1), row.getDouble(2))
  *
  * The SparseMatrix can be easily converted back to the MLlib distributed matricies using the same
  * type of casting methods.
  *
  * val coordinateMatrix: CoordinateMatrix = sparseMatrix.toCoordinateMatrix
  *
  * val blockMatrix: BlockMatrix = sparseMatrix.toBlockMatrix
  *
  * val localMatrix: LocalMatrix = sparseMatrix.toLocalMatrix
  *
  * @param entries Input data, a Dataset of MatrixEntry
  * @param nRows   number of rows, in case padding is needed
  * @param nCols   number of columns, in case padding is needed
  */
@Since("2.4.0")
class CoordinateMatrix(
    @Since("2.4.0") val entries: Dataset[MatrixEntry],
    private var nRows: Long,
    private var nCols: Long) extends DistributedMatrix {

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  @Since("2.4.0")
  def this(entries: Dataset[MatrixEntry]) = this(entries, 0L, 0L)

  /**
    * Get Number of Columns in Matrix
    *
    * @return
    */
  @Since("2.4.0")
  override def numCols(): Long = {
    if (nCols <= 0L) {
      computeSize()
    }
    nCols
  }

  /**
    * Get Number of Rows in Matrix
    *
    * @return
    */
  @Since("2.4.0")
  override def numRows(): Long = {
    if (nRows <= 0L) {
      computeSize()
    }
    nRows
  }

  /**
    * Transpose Matrix
    *
    * @return
    */
  @Since("2.4.0")
  def transpose(): CoordinateMatrix = {
    import entries.sparkSession.implicits._
    new CoordinateMatrix(entries.map(x => MatrixEntry(x.j, x.i, x.value)), nCols, nRows)
  }

  /**
    * Convert from Sparse to Dense format
    *
    * @return
    */
  @Since("2.4.0")
  def toIndexedRowMatrix(): IndexedRowMatrix = {
    import entries.sparkSession.implicits._

    new IndexedRowMatrix(
      vectorizeRows("index", "vector")
        .map(row => {
          val iVecSeq = row.getAs[mutable.WrappedArray[Row]](1).map(r => (r.getInt(0), r.getDouble(1)))
          val sparseVectorLength = iVecSeq.map(r => r._1).max + 1
          val iMLSparse = ml.linalg.Vectors.sparse(sparseVectorLength, iVecSeq)
          IndexedRow(row.getLong(0), iMLSparse)
        }))
  }

  /**
    * Converts to RowMatrix, dropping row indices after grouping by row index.
    * The number of columns must be within the integer range.
    */
  @Since("2.4.0")
  def toRowMatrix(): RowMatrix = {
    toIndexedRowMatrix.toRowMatrix()
  }

  /**
    * Converts to BlockMatrix. Creates blocks of `SparseMatrix` with size 1024 x 1024.
    */
  @Since("2.4.0")
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
  @Since("2.4.0")
  def toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")
    import entries.sparkSession.implicits._
    val m = numRows()
    val n = numCols()

    // Since block matrices require an integer row and col index
    require(math.ceil(m.toDouble / rowsPerBlock) <= Int.MaxValue,
      "Number of rows divided by rowsPerBlock cannot exceed maximum integer.")
    require(math.ceil(n.toDouble / colsPerBlock) <= Int.MaxValue,
      "Number of cols divided by colsPerBlock cannot exceed maximum integer.")

    val numRowBlocks = math.ceil(m.toDouble / rowsPerBlock).toInt
    val numColBlocks = math.ceil(n.toDouble / colsPerBlock).toInt
    val partitioner = GridPartitioner(numRowBlocks, numColBlocks, entries.rdd.partitions.length)

    val blocks: RDD[((Int, Int), Matrix)] = entries.rdd.map { entry =>
      val blockRowIndex = (entry.i / rowsPerBlock).toInt
      val blockColIndex = (entry.j / colsPerBlock).toInt

      val rowId = entry.i % rowsPerBlock
      val colId = entry.j % colsPerBlock

      ((blockRowIndex, blockColIndex), (rowId.toInt, colId.toInt, entry.value))
    }.groupByKey(partitioner).map { case ((blockRowIndex, blockColIndex), entry) =>
      val effRows = math.min(m - blockRowIndex.toLong * rowsPerBlock, rowsPerBlock.toLong).toInt
      val effCols = math.min(n - blockColIndex.toLong * colsPerBlock, colsPerBlock.toLong).toInt
      ((blockRowIndex, blockColIndex), SparseMatrix.fromCOO(effRows, effCols, entry))
    }

    new BlockMatrix(blocks.toDS(), rowsPerBlock, colsPerBlock, m, n)
  }

  /**
    * Compute the Rows and Columns of the Matrix
    * Used by the cols and rows getters
    */
  private def computeSize(): Unit = {
    // There may be empty columns at the very right and empty rows at the very bottom.
    import entries.sparkSession.implicits._
    val (m1, n1) = entries
      .map(entry => (entry.i, entry.j))
      .reduce((l1, l2) => (math.max(l1._1, l2._1), math.max(l1._2, l2._2)))

    nRows = math.max(nRows, m1 + 1L)
    nCols = math.max(nCols, n1 + 1L)
  }

  private[ml] def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  /**
    * Left multiplies this [[CoordinateMatrix]] to `other`, another [[CoordinateMatrix]]. The `colsPerBlock`
    * of this matrix must equal the `rowsPerBlock` of `other`.
    * Blocks with duplicate indices will be added with each other.
    *
    * @param other Matrix `B` in `A * B = C`
    *
    */
  def multiply(other: CoordinateMatrix): CoordinateMatrix = {
    import entries.sparkSession.implicits._

    val sparseVectorLength = numCols().toInt

    val dot = udf((iVec: mutable.WrappedArray[Row], jVec: mutable.WrappedArray[Row]) => {
      val iVecSeq = iVec.map(r => (r.getInt(0), r.getDouble(1)))
      val iMLSparse = V.sparse(sparseVectorLength, iVecSeq).toSparse
      val ibsv = new BSV[Double](iMLSparse.indices, iMLSparse.values, iMLSparse.size)

      val jVecSeq = jVec.map(r => (r.getInt(0), r.getDouble(1)))
      val jbdv = new BDV[Double](V.sparse(sparseVectorLength, jVecSeq).toArray)

      (jbdv.asDenseMatrix * ibsv).data(0)
    })

    val leftMatrixVectors: DataFrame = other.vectorizeCols("j", "jVec")

    new CoordinateMatrix(
      vectorizeRows("i", "iVec")
        .crossJoin(leftMatrixVectors)
        .withColumn("product", dot(col("iVec"), col("jVec")))
        .mapPartitions((row: Iterator[Row]) =>
          row.map(row =>
            MatrixEntry(
              row.getAs[Long]("i"),
              row.getAs[Long]("j"),
              row.getAs[Double]("product"))))
    )
  }

  private[distributed] def vectorizeRows(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.i, (d.j.toInt, d.value)))
      .groupBy($"_1").agg(collect_list($"_2"))
      .withColumnRenamed("_1", "i")
      .withColumnRenamed("collect_list(_2)", colName)
  }

  private[distributed] def vectorizeCols(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.j, (d.i.toInt, d.value)))
      .groupBy($"_1").agg(collect_list($"_2"))
      .withColumnRenamed("_1", "j")
      .withColumnRenamed("collect_list(_2)", colName)
  }

}
