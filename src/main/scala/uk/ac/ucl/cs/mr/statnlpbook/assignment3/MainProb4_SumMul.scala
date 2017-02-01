package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}

import scala.collection.mutable

object MainProb4_SumMul extends App {
  //-----------------------------------------------
  // Settings
  //-----------------------------------------------
  val runGridSearch = false

  // Settings for single run
  var learningRate = 1e-3
  var vectorRegularizationStrength = 0.30
  var matrixRegularizationStrength = 0.0
  var wordDim = 30
  var hiddenSize = 10
  var numEpoch = 100
  var dropoutProb = 0.5

  // best hyperparameters for problem2 - find by grid search
  //var learningRate = 0.01
  //var wordDim = 10
  //var vectorRegularizationStrength = 0.1
  //var numEpoch = 23

  // Settings for Grid search
  val wordDimList = List(10, 20)
  val hiddenDimList = List(10, 20)
  val vecRegStrList = List(0.1, 1e-2, 1e-3)
  val learningRateList = List(0.01, 0.001)
  
  //-----------------------------------------------

  val trainSetName = "train"
  val validationSetName = "dev"
  val testSetName = "test"
  val gridSearchResults = new mutable.HashMap[String, Double]()
  var learningRateCounter = 0
  var model: Model =
//    new RecurrentNeuralNetworkWithDropoutModel(wordDim, hiddenSize, vectorRegularizationStrength,
//      matrixRegularizationStrength, dropoutProb)
    new SumMultOfWordVectorsModel(wordDim, vectorRegularizationStrength)

  if (runGridSearch){
    gridSearch()
  }else{
    StochasticGradientDescentLearner2(model, trainSetName, numEpoch, learningRate, epochHook)
    predictAndWrite(model, testSetName)
  }

  def epochHook(iter: Int, accLoss: Double, currLearningRate: Double): Double = {
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
    }
    else {
      val trainAcc = 100 * Evaluator(model, trainSetName)
      val devAcc = 100 * Evaluator(model, validationSetName)


      println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f\tLearning rate %1.5f".format(
        iter, accLoss, trainAcc, devAcc, currLearningRate))
      if (trainAcc - devAcc > 2.0 && learningRateCounter == 0){
          learningRateCounter = 1
          return currLearningRate / 10
      }
    }
    currLearningRate
  }

  def predictAndWrite (model: Model, corpus: String) = {
    val writer = new PrintWriter(new File("predictions.txt"))
    writer.write("prediction\n")
    val total = SentimentAnalysisCorpus.numExamples(corpus)
    for (i <- 0 until total) {
      val (sentence, target) = SentimentAnalysisCorpus.test(i)
      val predict = model.predict(sentence)
      if (predict == false) writer.append("0\n")
      else writer.append("1\n")
    }
    writer.close()
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
            StochasticGradientDescentLearner2(model, trainSetName, numEpoch, learningRate, epochHook)
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