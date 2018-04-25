// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, norm => brzNorm, svd => brzSvd}
import com.microsoft.ml.spark.RecommendationTestBase
import org.apache.spark.ml.linalg.{Vector, Vectors}

class RowMatrixSuite extends RecommendationTestBase {

  val m = 4
  val n = 3
  val arr = Array(0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 0.0, 2.0, 5.0, 8.0, 1.0)
  val denseData = Seq(
    Vectors.dense(0.0, 1.0, 2.0),
    Vectors.dense(3.0, 4.0, 5.0),
    Vectors.dense(6.0, 7.0, 8.0),
    Vectors.dense(9.0, 0.0, 1.0)
  )
  val sparseData = Seq(
    Vectors.sparse(3, Seq((1, 1.0), (2, 2.0))),
    Vectors.sparse(3, Seq((0, 3.0), (1, 4.0), (2, 5.0))),
    Vectors.sparse(3, Seq((0, 6.0), (1, 7.0), (2, 8.0))),
    Vectors.sparse(3, Seq((0, 9.0), (2, 1.0)))
  )

  val principalComponents = BDM(
    (0.0, 1.0, 0.0),
    (math.sqrt(2.0) / 2.0, 0.0, math.sqrt(2.0) / 2.0),
    (math.sqrt(2.0) / 2.0, 0.0, -math.sqrt(2.0) / 2.0))
  val explainedVariance = BDV(4.0 / 7.0, 3.0 / 7.0, 0.0)

  var denseMat: RowMatrix = _
  var sparseMat: RowMatrix = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    import session.implicits._
    denseMat = new RowMatrix(sc.parallelize(denseData, 2).map(RowVector(_)).toDS())
    sparseMat = new RowMatrix(sc.parallelize(sparseData, 2).map(RowVector(_)).toDS())
  }

  test("size") {
    assert(denseMat.numRows() === m)
    assert(denseMat.numCols() === n)
    assert(sparseMat.numRows() === m)
    assert(sparseMat.numCols() === n)
  }

  test("empty rows") {
    import session.implicits._
    val rows = sc.parallelize(Seq[Vector](), 1).map(RowVector(_)).toDS()
    val emptyMat = new RowMatrix(rows)
    intercept[RuntimeException] {
      emptyMat.numCols()
    }
    intercept[RuntimeException] {
      emptyMat.numRows()
    }
  }

}
