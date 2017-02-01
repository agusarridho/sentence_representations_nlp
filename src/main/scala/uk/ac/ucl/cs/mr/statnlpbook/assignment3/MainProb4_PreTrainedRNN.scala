package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable

object MainProb4_PreTrainedRNN extends App {
  //-----------------------------------------------
  // Settings
  //-----------------------------------------------
  val runGridSearch = true

  //Single run
  var learningRate = 0.01
  var vectorRegularizationStrength = 0.01
  var matrixRegularizationStrength = 0.0
  var wordDim = 10
  var hiddenDim = 20 //You can try dimensions 10, 20, 30, 40, 50. I generated files for these
  var numEpoch = 25

  //Best parameters
//  var learningRate = 1e-3
//  var vectorRegularizationStrength = 0.0
//  var matrixRegularizationStrength = 0.0
//  var wordDim = 10
//  var hiddenDim = 10
//  var numEpoch = 50

  // Grid search settings

  // Grid search MIROSLAV
  val wordDimList = List(10)
  val hiddenDimList = List(10)
  val vecRegStrList = List(0.1, 1e-2)
  val matRegStrList = List(0.1, 1e-2)
  val learningRateList = List(0.1, 0.01)

  // Grid search AGUS
//  val wordDimList = List(15, 20)
//  val hiddenDimList = List(20)
//  val vecRegStrList = List(1e-2, 1e-3)
//  val matRegStrList = List(1e-2, 1e-3)
//  val learningRateList = List(1e-2, 1e-3)

  // Grid search VICTOR
//  val wordDimList = List(20, 25)
//  val hiddenDimList = List(30)
//  val vecRegStrList = List(e-3, 1e-4)
//  val matRegStrList = List(1e-3, 1e-4)
//  val learningRateList = List(1e-3, 1e-4)
  //--------------------------------------------------

  val trainSetName = "train"
  val validationSetName = "dev"
  val gridSearchResults = new mutable.HashMap[String, Double]()
  var model: Model = new PreTrainedRecurrentNeuralNetworkModel(wordDim, hiddenDim,
    vectorRegularizationStrength, matrixRegularizationStrength,
    s"trainedVectors_Dim_${hiddenDim}.txt")

  if (runGridSearch){
    gridSearch()
  }else{
    StochasticGradientDescentLearner(model, trainSetName, numEpoch, learningRate, epochHook)
  }

  def epochHook(iter: Int, accLoss: Double): Unit = {
    if (runGridSearch) {
      if (iter == numEpoch - 1) {
        val trainAcc = 100 * Evaluator(model, trainSetName)
        val devAcc = 100 * Evaluator(model, validationSetName)
        val iterString = s"Epoch ${iter} | Word Size ${wordDim} | Hidden Size ${hiddenDim} | " +
          s"Learning Rate ${learningRate} | Vector Reg ${vectorRegularizationStrength} | " +
          s"Matrix Reg ${matrixRegularizationStrength} | " +
          s"Loss ${accLoss} | Train Acc ${trainAcc} | Dev Acc ${devAcc}"

        gridSearchResults(iterString) = trainAcc + devAcc
        println(iterString + " " + trainAcc + devAcc)
      }
      else{
        println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
          iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
      }
    }
  }

  
  def gridSearch(): Unit = {
    wordDimList.foreach(aDim => {
      hiddenDimList.foreach(aHidDim => {
        vecRegStrList.foreach(aVecRegStrength => {
          matRegStrList.foreach(aMatRegStrength => {
            learningRateList.foreach(aLearningRate => {
              val keys = LookupTable.trainableWordVectors.keySet
              keys.foreach(k => LookupTable.trainableWordVectors.remove(k))
              wordDim = aDim
              hiddenDim = aHidDim
              vectorRegularizationStrength = aVecRegStrength
              matrixRegularizationStrength = aMatRegStrength
              learningRate = aLearningRate
              model = new PreTrainedRecurrentNeuralNetworkModel(
                wordDim, hiddenDim, vectorRegularizationStrength,
                matrixRegularizationStrength, s"trainedVectors_Dim_${hiddenDim}.txt")
              StochasticGradientDescentLearner(model, trainSetName, numEpoch, learningRate, epochHook)
            })
          })
        })
      })
    })

    gridSearchResults.foreach(pair => {
      //println(pair._1 + ": " + pair._2)
    })
    val max = gridSearchResults.max
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