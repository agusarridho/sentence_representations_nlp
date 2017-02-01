package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}

import scala.collection.mutable

/**
 * @author rockt
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */
  //Original variables given to us - original values of hyperparameters
  val learningRate = 1e-4
  val vectorRegularizationStrength = 1e-3
  val matrixRegularizationStrength = 0.0
  val wordDim = 10
  val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"

  // Lists of values prepared for grid search
  val problem2_wordDim = List(10, 20)
  val problem2_hiddenDim = List(10, 20)
  val problem2_vecRegStr = List(0.1, 1e-2, 1e-3)
  val problem2_matRegStr = List(0.1, 1e-2, 1e-3)
  val problem2_learningRate = List(0.01, 0.001)
  val numEpoch = 50

  // Initialize model for Problem 2
  var myModel: Model = new SumOfWordVectorsModel(10, 0.01)

  //  // Initialize model for Problem 3
  //  var myModelRNN: Model = new RecurrentNeuralNetworkModel(10,10,0.01,0.01)

  val sumOfWordResults = new mutable.HashMap[String, Double]()
  var currWord = 0
  var currHidden = 0
  var currVecRegStr = 0.0
  var currMatRegStr = 0.0
  var currLearningRate = 0.0

  // best hyperparameters for problem2 - find by grid search
  val bestLearningRate2 = 0.01
  val bestWordSize2 = 10
  val bestRegularizationParameter2 = 0.1
  val bestNumberOfEpochs2 = 23

  //    Original code given to us
  //  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //  val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
  //val model: Model = new PreTrainedSumOfWordVectorsModel(wordDim, vectorRegularizationStrength,
    //"trainVectors.txt")
  val model: Model = new SumOfWordVectorsModel(bestWordSize2, bestRegularizationParameter2)
  StochasticGradientDescentLearner(model, trainSetName, 1, bestLearningRate2, epochHook)

  def epochHook(iter: Int, accLoss: Double): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }

  val pw = new PrintWriter(new File("words.txt"))
  val pw2 = new PrintWriter(new File("vectorRepresentation.txt"))
  val pw3 = new PrintWriter(new File("score.txt"))
  val keySet = model.vectorParams.keySet
  keySet.foreach{
    k => pw.write(k+"\n")
      val vectorRepr = model.vectorParams(k).output.toString()
      pw2.write(vectorRepr.substring(12, vectorRepr.length()-2)+"\n")
      val wordVector = model.vectorParams(k).output
      // this is not working, it should return score, but returns only zeros.
      val score = model.scoreSentence(wordVector).output
      pw3.write(score + "\n")
  }
  pw.close
  pw2.close
  pw3.close

  // epoch hook for Problem 2
  //  def epochHook(iter: Int, accLoss: Double): Unit = {
  //    // to print and add only the result of last iteration
  //    if (!accLoss.isNaN){
  //      val trainAcc = 100 * Evaluator(myModel, trainSetName)
  //      val devAcc = 100 * Evaluator(myModel, validationSetName)
  //      val iterString = s"Epoch ${iter} | Word Size ${currWord} | Learning Rate ${currLearningRate} | " +
  //        s"Regularization ${currVecRegStr} | Loss $accLoss | Train Acc ${trainAcc} | Dev Acc $devAcc"
  //      sumOfWordResults(iterString) = devAcc
  //      println(iterString)
  //    }
  //  }

  //  // epoch hook for Problem 3
  //  def epochHookRNN(iter: Int, accLoss: Double): Unit = {
  //    // to print and add only the result of last iteration
  //    if (iter == numEpoch - 1){
  //      val trainAcc = 100 * Evaluator(myModelRNN, trainSetName)
  //      val devAcc = 100 * Evaluator(myModelRNN, validationSetName)
  //      val iterString = s"Epoch ${iter} | Word Size ${currWord} | Hidden Size ${currHidden} | " +
  //        s"Learning Rate ${currLearningRate} | Vector Reg ${currVecRegStr} | Matrix Reg ${currMatRegStr} | " +
  //        s"Loss ${accLoss} | Train Acc ${trainAcc} | Dev Acc ${devAcc}"
  //
  //      sumOfWordResults(iterString) = trainAcc + devAcc
  //      println(iterString + " " + trainAcc + devAcc)
  //    }
  //  }

  // grid search for Problem 2
  //  def gridSearchSumOfWord(): Unit = {
  //    problem2_wordDim.foreach(aDim => {
  //      problem2_vecRegStr.foreach(aRegStrength => {
  //        problem2_learningRate.foreach(aLearningRate => {
  //          val keys = LookupTable.trainableWordVectors.keySet
  //          keys.foreach(k => LookupTable.trainableWordVectors.remove(k))
  //          currWord = aDim
  //          currLearningRate = aLearningRate
  //          currVecRegStr = aRegStrength
  //          myModel = new SumOfWordVectorsModel(currWord, currVecRegStr)
  //          StochasticGradientDescentLearner(myModel, trainSetName, numEpoch, currLearningRate, epochHook)
  //        })
  //      })
  //    })
  //    sumOfWordResults.foreach(pair => {
  //      //println(pair._1 + ": " + pair._2)
  //    })
  //    val max = sumOfWordResults.maxBy(_._2)
  //    println(s"Best config is ${max._1}")
  //  }
  //  gridSearchSumOfWord()

  //  // grid search for Problem 3
  //  def gridSearchRNN(): Unit = {
  //    problem2_wordDim.foreach(aDim => {
  //      problem2_hiddenDim.foreach(aHidDim => {
  //        problem2_vecRegStr.foreach(aVecRegStrength => {
  //          problem2_matRegStr.foreach(aMatRegStrength => {
  //            problem2_learningRate.foreach(aLearningRate => {
  //              val keys = LookupTable.trainableWordVectors.keySet
  //              keys.foreach(k => LookupTable.trainableWordVectors.remove(k))
  //              currWord = aDim
  //              currHidden = aHidDim
  //              currVecRegStr = aVecRegStrength
  //              currMatRegStr = aMatRegStrength
  //              currLearningRate = aLearningRate
  //              myModelRNN = new RecurrentNeuralNetworkModel(currWord, currHidden, currVecRegStr, currMatRegStr)
  //              StochasticGradientDescentLearner(myModel, trainSetName, numEpoch, currLearningRate, epochHook)
  //            })
  //          })
  //        })
  //      })
  //    })
  //
  //    sumOfWordResults.foreach(pair => {
  //      //println(pair._1 + ": " + pair._2)
  //    })
  //    val max = sumOfWordResults.max
  //    println(s"Best config is ${max._1}")
  //  }
  //  gridSearchRNN()

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