// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import com.microsoft.ml.spark.RecommendationTestBase

class IndexedRowMatrixSpec extends RecommendationTestBase {

  test("Dense Matrix Test") {
    def testDense(row: Int, col: Int): Unit = {
      val xds = new CoordinateMatrix(makeMatrixDF(row, col)).toIndexedRowMatrix

      assert(xds.numRows() == row)
      assert(xds.numCols() == col)
      assert(xds.rows.count() == xds.numRows())
      ()
    }

    testDense(5, 5)
    testDense(10, 5)
    testDense(5, 10)
  }

}
