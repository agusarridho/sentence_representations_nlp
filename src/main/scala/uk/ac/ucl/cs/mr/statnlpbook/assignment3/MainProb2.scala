package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{PrintWriter, File}

import scala.collection.mutable

object MainProb2 extends App {
  //-----------------------------------------------
  // Settings
  //-----------------------------------------------
  val runGridSearch = false

  // Settings for single run
//  var learningRate = 1e-3
//  var vectorRegularizationStrength = 0.0
//  var wordDim = 10
//  var numEpoch = 50

  // best hyperparameters for problem2 - find by grid search
  var learningRate = 0.01
  var wordDim = 10
  var vectorRegularizationStrength = 0.1
  var numEpoch = 50

  // Settings for Grid search
  val wordDimList = List(10, 20)
  val hiddenDimList = List(10, 20)
  val vecRegStrList = List(0.1, 1e-2, 1e-3)
  val learningRateList = List(0.01, 0.001)
  
  //-----------------------------------------------

  val trainSetName = "train"
  val validationSetName = "dev"
  val gridSearchResults = new mutable.HashMap[String, Double]()
  var model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)

  if (runGridSearch){
    gridSearch()
  }else{
    StochasticGradientDescentLearner(model, trainSetName, numEpoch, learningRate, epochHook)
    printToFiles(model)
  }

  def epochHook(iter: Int, accLoss: Double): Unit = {
    if (runGridSearch){
      // to print and add only the result of last iteration
      if (!accLoss.isNaN){
        val trainAcc = 100 * Evaluator(model, trainSetName)
        val devAcc = 100 * Evaluator(model, validationSetName)
        val iterString = s"Epoch ${iter} | Word Size ${wordDim} | Learning Rate ${learningRate} | " +
          s"Regularization ${vectorRegularizationStrength} | Loss $accLoss | Train Acc ${trainAcc} | Dev Acc $devAcc"
        gridSearchResults(iterString) = devAcc
        println(iterString)
      }
    }else {
      println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
        iter, accLoss, 100 * Evaluator(model, trainSetName), 100 * Evaluator(model, validationSetName)))
    }
  }

  def printToFiles(aModel: Model) = {
//    val pw = new PrintWriter(new File("words.txt"))
    val pw2 = new PrintWriter(new File("vectorRepresentation.txt"))
    pw2.write("word\t x1\t x2\t x3\t x4\t x5\t x6\t x7\t x8\t x9\t x10 \n")
    val pw3 = new PrintWriter(new File("score.txt"))
    pw3.write("word\t score \n")
    val keySet = aModel.vectorParams.keySet.take(11000).toSet
    keySet.foreach {
      k =>
        if (!k.contains("\t") | k.equals("")){
          val vectorRepr = aModel.vectorParams(k).output
          val score = aModel.scoreSentence(vectorRepr).forward()
          if (score > 0.01 & k.length >= 1 & !vectorRepr.toString().contains("E-")){
            pw2.append(k + "\t " + vectorRepr.toString().replace(",", "\t").substring(12, vectorRepr.toString().replace(",", "\t").length() - 2) + "\n")
            pw3.append(k+ "\t "+score + "\n")
          }
        }
    }
//    pw.close
    pw2.close
    pw3.close
  }

   //grid search for Problem 2
    def gridSearch(): Unit = {
      wordDimList.foreach(aDim => {
        vecRegStrList.foreach(aRegStrength => {
          learningRateList.foreach(aLearningRate => {
            val keys = LookupTable.trainableWordVectors.keySet
            keys.foreach(k => LookupTable.trainableWordVectors.remove(k))
            wordDim = aDim
            learningRate = aLearningRate
            vectorRegularizationStrength = aRegStrength
            model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
            StochasticGradientDescentLearner(model, trainSetName, numEpoch, learningRate, epochHook)
          })
        })
      })
      gridSearchResults.foreach(pair => {
        //println(pair._1 + ": " + pair._2)
      })
      val max = gridSearchResults.maxBy(_._2)
      println(s"Best config is ${max._1}")
    }


  /**
   * Comment this in if you want to look at trained parameters
   */
  /*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  */
}