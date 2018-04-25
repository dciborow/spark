// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import com.microsoft.ml.spark.RecommendationTestBase
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.distributed
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.Dataset

import scala.language.implicitConversions

class CoordinateMatrixSpec extends RecommendationTestBase {

  /**
    * Test multiplication of...
    * ...two square matrices
    * ...one rectangular, and one square matrix
    */
  test("Mat Mul test") {
    def testMatrixMultiplication(row: Int, col: Int): Unit = {
      implicit def dsToSM(ds: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(ds)

      val xds = makeMatrixDF(row, col)
      assert(xds.numRows() == row)
      assert(xds.numCols() == col)

      val yds = makeMatrixDF(col, col)
      assert(yds.numCols() == col)
      assert(yds.numRows() == col)

      val product = xds multiply yds
      assert(product.numCols == col)
      assert(product.numRows == row)

      val rddProduct = xds.toMLLibBlockMatrix multiply yds.toMLLibBlockMatrix
      assert(product.toMLLibLocalMatrix == rddProduct.toLocalMatrix().asML)
      ()
    }

    //Test Square Matrix
    testMatrixMultiplication(10, 10)

    //Test Rectangle
    testMatrixMultiplication(5, 10)
  }

  /**
    * 5k x 5k
    * DS RDD: 36754
    * DS: 42801
    * RDD: 20778
    *
    * 1.1k x 1.1k
    * DS RDD: 7020
    * DS: 4454
    * RDD: 6895
    *
    */
  test("To BlockMatrix Performance") {
    val loop = 5
    val row = 1100
    val col = 1100
    val df = makeMatrixDF(row, col).cache
    assert(df.count > 0)
    val xds = new CoordinateMatrix(df, row.toLong, col.toLong) //.sample(false, 0.01)

    val start = System.currentTimeMillis()
    (0 to loop).foreach(_ => {
      assert(xds.toBlockMatrix().blocks.count() > 0)
    })
    val end = System.currentTimeMillis()
    val timeDSRDD = (end - start) / loop
    println("DS RDD: " + ((end - start) / loop))

    val start3 = System.currentTimeMillis()
    (0 to loop).foreach(_ => {
      assert(xds.toBlockMatrix(rdd = false).blocks.count() > 0)
    })
    val end3 = System.currentTimeMillis()
    val timeDS = (end3 - start3) / loop
    println("DS: " + ((end3 - start3) / loop))

    val xrdd = new distributed.CoordinateMatrix(df.rdd, row.toLong, col.toLong)
    val start2 = System.currentTimeMillis()
    (0 to loop).foreach(_ => {
      assert(xrdd.toBlockMatrix().blocks.count() > 0)
    })
    val end2 = System.currentTimeMillis()
    val timeRdd = (end2 - start2) / loop
    println(("RDD: " + ((end2 - start2) / loop)))
  }

  test("To Sparse") {
    def testBlockMatrixMultiplication(row: Int, col: Int): Unit = {
      implicit def dsToSM(ds: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(ds)

      val (rddOn, dsRddOn, dsOn, coordOn) = (false, false, true, false)
      val loop = 5

      val xds = new CoordinateMatrix(makeMatrixDF(row, col), row.toLong, col.toLong)
      //.sample(false, 0.01)
      val xdsBlock = xds.toBlockMatrix()
      //      assert(xds.numRows() == row)
      assert(xds.numCols() == col)

      val yds = makeMatrixDF(col, col)
      val ydsBlock = yds.toBlockMatrix()
      assert(yds.numCols() == col)
      assert(yds.numRows() == col)

      assert(xdsBlock.cache().blocks.count() > 0)
      assert(ydsBlock.cache().blocks.count() > 0)

      val rddLeft = xds.toMLLibBlockMatrix
      val rddRight = yds.toMLLibBlockMatrix

      assert(rddLeft.cache().blocks.count() > 0)
      assert(rddRight.cache().blocks.count() > 0)

      if (rddOn) {
        assert((rddLeft multiply rddRight).blocks.count > 0)

        val start2 = System.currentTimeMillis()
        (0 to loop).foreach(i => {
          val product = rddLeft multiply rddRight
          assert(product.blocks.count > 0)
          ()
        }
        )
        val end2 = System.currentTimeMillis()
        println("Duration RDD: " + ((end2 - start2) / loop))
      }

      if (dsRddOn) {
        assert((xdsBlock multiply ydsBlock).blocks.count > 0)

        val start = System.currentTimeMillis()
        (0 to loop).foreach(i => {
          val product = xdsBlock multiply ydsBlock
          assert(product.blocks.count > 0)
          ()
        }
        )
        val end = System.currentTimeMillis()
        println("Duration DS RDD: " + ((end - start) / loop))
      }

      if (dsOn) {
        val xdsBlock2 = xds.toBlockMatrix(false)
        val ydsBlock2 = yds.toBlockMatrix(false)
        assert(xdsBlock2.cache().blocks.count() > 0)
        assert(ydsBlock2.cache().blocks.count() > 0)

        assert((xdsBlock2.multiply(ydsBlock2, rdd = false)).blocks.count > 0)

        val start3 = System.currentTimeMillis()
        (0 to loop).foreach(i => {
          val product = (xdsBlock2.multiply(ydsBlock2, rdd = false))
          assert(product.blocks.count > 0)
          ()
        }
        )
        val end3 = System.currentTimeMillis()

        println("Duration DS: " + ((end3 - start3) / loop))
      }

      if (coordOn) {
        assert((xds.multiply(yds)).entries.count() > 0)

        val start4 = System.currentTimeMillis()
        (0 to loop).foreach(i => {
          val product = (xds.multiply(yds))
          assert(product.entries.count > 0)
          ()
        }
        )
        val end4 = System.currentTimeMillis()

        println("Duration Coord Matrix DS: " + ((end4 - start4) / loop))
      }
      ()
    }

    //Test Square Matrix
    //    testBlockMatrixMultiplication(100000, 10)

    //Test Rectangle
    testBlockMatrixMultiplication(2000, 200)

  }

  /**
    * Test Transpose of...
    * ...a transpose is reversible
    * ...a matrix transpose times the original matrix
    */
  test("Rectangle Transpose mat mul test") {
    implicit def dsToSM(ds: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(ds)

    val row = 10
    val col = 5

    val xds = makeMatrixDF(row, col)
    assert(xds.numRows() == row)
    assert(xds.numCols() == col)
    assert(xds.numCols() == xds.transpose().transpose().numCols())
    assert(xds.numRows() === xds.transpose().transpose().numRows())
    assert(xds.collect() === xds.transpose().transpose().entries.collect())

    val product = xds.transpose multiply xds
    assert(product.numCols == col)
    assert(product.numRows == col)

    val rddProduct = xds.toMLLibBlockMatrix.transpose multiply xds.toMLLibBlockMatrix
    assert(product.toMLLibLocalMatrix == rddProduct.toLocalMatrix.asML)
  }

  /**
    * Test multiplication of...
    * ...a dataframe with duplicate elements
    */
  ignore("Duplicates mat mul test") {
    implicit def dsToSM(ds: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(ds)

    val row = 5
    val col = 10

    val xds = makeMatrixDF(row, col)
    val yds = makeMatrixDF(col, col)

    val product = (xds.union(xds) multiply yds.union(yds)).toMLLibLocalMatrix
    assert(product.numCols == col)
    assert(product.numRows == row)

    val rddProduct = (xds.union(xds).toMLLibBlockMatrix multiply yds.union(yds).toMLLibBlockMatrix).toLocalMatrix.asML
    assert(product == rddProduct)
  }
}
