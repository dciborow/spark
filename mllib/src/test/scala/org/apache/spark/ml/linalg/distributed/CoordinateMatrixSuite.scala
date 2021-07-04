// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM}
import com.microsoft.ml.spark.RecommendationTestBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.MatrixEntry

class CoordinateMatrixSuite extends RecommendationTestBase {
  val m = 5
  val n = 4
  var mat: CoordinateMatrix = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    import session.implicits._
    val entries = session.createDataset(
      Seq(
        (0, 0, 1.0),
        (0, 1, 2.0),
        (1, 1, 3.0),
        (1, 2, 4.0),
        (2, 2, 5.0),
        (2, 3, 6.0),
        (3, 0, 7.0),
        (3, 3, 8.0),
        (4, 1, 9.0)).map { case (i, j, value) =>
        MatrixEntry(i.toLong, j.toLong, value)
      })
    mat = new CoordinateMatrix(entries)
  }

  test("size") {
    assert(mat.numRows() === m)
    assert(mat.numCols() === n)
  }

  test("empty entries") {
    import session.implicits._
    val entries = session.createDataset(Seq[MatrixEntry]())
    val emptyMat = new CoordinateMatrix(entries)
    intercept[RuntimeException] {
      emptyMat.numCols()
    }
    intercept[RuntimeException] {
      emptyMat.numRows()
    }
  }

  test("toBreeze") {
    val expected = BDM(
      (1.0, 2.0, 0.0, 0.0),
      (0.0, 3.0, 4.0, 0.0),
      (0.0, 0.0, 5.0, 6.0),
      (7.0, 0.0, 0.0, 8.0),
      (0.0, 9.0, 0.0, 0.0))
    assert(mat.toBreeze() === expected)
  }

  test("transpose") {
    val transposed = mat.transpose()
    assert(mat.toBreeze().t === transposed.toBreeze())
  }

  ignore("toIndexedRowMatrix") {
    val indexedRowMatrix = mat.toIndexedRowMatrix()
    val expected = BDM(
      (1.0, 2.0, 0.0, 0.0),
      (0.0, 3.0, 4.0, 0.0),
      (0.0, 0.0, 5.0, 6.0),
      (7.0, 0.0, 0.0, 8.0),
      (0.0, 9.0, 0.0, 0.0))
    assert(indexedRowMatrix.toBreeze() === expected)
  }

  ignore("toRowMatrix") {
    val rowMatrix = mat.toRowMatrix()
    val rows = rowMatrix.rows.collect().toSet
    val expected = Set(
      Vectors.dense(1.0, 2.0, 0.0, 0.0),
      Vectors.dense(0.0, 3.0, 4.0, 0.0),
      Vectors.dense(0.0, 0.0, 5.0, 6.0),
      Vectors.dense(7.0, 0.0, 0.0, 8.0),
      Vectors.dense(0.0, 9.0, 0.0, 0.0))
    assert(rows === expected)
  }

  test("toBlockMatrixMatrix") {
    val blockMat = mat.toBlockMatrix(2, 2)
    assert(blockMat.numRows() === m)
    assert(blockMat.numCols() === n)
    assert(blockMat.toBreeze() === mat.toBreeze())

    intercept[IllegalArgumentException] {
      mat.toBlockMatrix(-1, 2)
    }
    intercept[IllegalArgumentException] {
      mat.toBlockMatrix(2, 0)
    }
  }

}
