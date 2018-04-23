// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import com.microsoft.ml.spark.RecommendationTestBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import breeze.linalg.{diag => brzDiag, DenseMatrix => BDM, DenseVector => BDV}

class IndexedRowMatrixSuite extends RecommendationTestBase{

  val m = 4
  val n = 3
  val data: Seq[IndexedRow] = Seq(
    (0L, Vectors.dense(0.0, 1.0, 2.0)),
    (1L, Vectors.dense(3.0, 4.0, 5.0)),
    (3L, Vectors.dense(9.0, 0.0, 1.0))
  ).map(x => IndexedRow(x._1, x._2))
  var indexedRows: Dataset[IndexedRow] = _

  override def beforeAll() {
    super.beforeAll()
    import session.implicits._
    indexedRows = session.createDataset(data)
  }

  test("size") {
    val mat1 = new IndexedRowMatrix(indexedRows)
    assert(mat1.numRows() === m)
    assert(mat1.numCols() === n)

    val mat2 = new IndexedRowMatrix(indexedRows, 5, 0)
    assert(mat2.numRows() === 5)
    assert(mat2.numCols() === n)
  }

  test("empty rows") {
    import session.implicits._
    val rows = session.createDataset(Seq[IndexedRow]())
    val mat = new IndexedRowMatrix(rows)
    intercept[RuntimeException] {
      mat.numRows()
    }
    intercept[RuntimeException] {
      mat.numCols()
    }
  }

  test("toBreeze") {
    val mat = new IndexedRowMatrix(indexedRows)
    val expected = BDM(
      (0.0, 1.0, 2.0),
      (3.0, 4.0, 5.0),
      (0.0, 0.0, 0.0),
      (9.0, 0.0, 1.0))
    assert(mat.toBreeze() === expected)
  }

  ignore("toRowMatrix") {
    val idxRowMat = new IndexedRowMatrix(indexedRows)
    val rowMat = idxRowMat.toRowMatrix()
    assert(rowMat.numCols() === n)
    assert(rowMat.numRows() === 3, "should drop empty rows")
    assert(rowMat.rows.collect().toSeq === data.map(_.vector).toSeq)
  }

  test("toCoordinateMatrix") {
    val idxRowMat = new IndexedRowMatrix(indexedRows)
    val coordMat = idxRowMat.toCoordinateMatrix()
    assert(coordMat.numRows() === m)
    assert(coordMat.numCols() === n)
    assert(coordMat.toBreeze() === idxRowMat.toBreeze())
  }

}
