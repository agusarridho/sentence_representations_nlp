package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable
import scala.io.Source

/**
 * @author rockt
 */
object LookupTable {
  val fixedWordVectors = new mutable.HashMap[String, VectorConstant]()
  val trainableWordVectors = new mutable.HashMap[String, VectorParam]()

  def addFixedWordVector(word: String, vector: Vector): Block[Vector] = {
    fixedWordVectors.getOrElseUpdate(word, vector)
  }

  def addTrainableWordVector(word: String, dim: Int = 300): Block[Vector] = {
    trainableWordVectors.getOrElseUpdate(word, VectorParam(dim))
  }

  def addTrainableWordVector(word: String, vector: Vector): Block[Vector] = {
    val param = VectorParam(vector.activeSize)
    param.set(vector)
    trainableWordVectors.getOrElseUpdate(word, param)
  }

  def get(word: String): Block[Vector] = trainableWordVectors.getOrElse(word, fixedWordVectors(word))

  def getFixedOrTrained (word: String, wordSize: Int): Block[Vector] = {
    if (fixedWordVectors.contains(word))
      fixedWordVectors(word)
    else
      addTrainableWordVector(word, wordSize)
  }

  def loadPreTrainedWordVectors(file: String, vectorSize: Int) = {
    for (line <- io.Source.fromFile(s"./data/assignment3/${file}", "ISO-8859-1").getLines()) {
      val fields = line.split(" ")
      if (fields.length >= vectorSize){
        addFixedWordVector(fields(0), vec(fields.drop(1).map(_.toDouble): _*))
        //addTrainableWordVector(fields(0), vec(fields.drop(1).map(_.toDouble): _*))
      }
    }
  }

  def cleanTrainableVector(): Unit ={
    trainableWordVectors.keySet.foreach(key => {
      trainableWordVectors.remove(key)
    })
  }
}
