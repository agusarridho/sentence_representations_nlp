package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{QuasiTensor, TensorLike, sum}
import breeze.numerics._

/**
 * Problem 1
 */
object GradientChecker extends App {
  val EPSILON = 1e-6

  /**
   * For an introduction see http://cs231n.github.io/neural-networks-3/#gradcheck
   *
   * This is a basic implementation of gradient checking.
   * It is restricted in that it assumes that the function to test evaluates to a double.
   * Moreover, another constraint is that it always tests by backpropagating a gradient of 1.0.
   */
  def apply[P](model: Block[Double], paramBlock: ParamBlock[P]) = {
    paramBlock.resetGradient()
    model.forward()
    model.backward(1.0)

    var avgError = 0.0

    val gradient = paramBlock.gradParam match {
      case m: Matrix => m.toDenseVector
      case v: Vector => v
    }

    /**
     * Calculates f_theta(x_i + eps)
     * @param index i in x_i
     * @param eps value that is added to x_i
     * @return
     */
    def wiggledForward(index: Int, eps: Double): Double = {
      var result = 0.0
      paramBlock.param match {
        case v: Vector =>
          val tmp = v(index)
          v(index) = tmp + eps
          result = model.forward()
          v(index) = tmp
        case m: Matrix =>
          val (row, col) = m.rowColumnFromLinearIndex(index)
          val tmp = m(row, col)
          m(row, col) = tmp + eps
          result = model.forward()
          m(row, col) = tmp
      }
      result
    }

    for (i <- 0 until gradient.activeSize) {
      //todo: your code goes here!
      val gradientExpected: Double =
        (wiggledForward(i, EPSILON) - wiggledForward(i, -EPSILON)) / (2 * EPSILON)

      avgError = avgError + math.abs(gradientExpected - gradient(i))

      assert(
        math.abs(gradientExpected - gradient(i)) < EPSILON,
        "Gradient check failed!\n" +
          s"Expected gradient for ${i}th component in input is $gradientExpected but I got ${gradient(i)}"
      )
    }

    println("    Average error: " + avgError)
  }

  /**
    * A very silly block to test if gradient checking is working.
    * Will only work if the implementation of the Dot block is already correct
    */



  //Dot and Sigmoid tests
  val a_sigmoid = vec(-1.5, 3.0, 1.5, 0.5)
  val paramBlock_sigmoid = VectorParam(4)
  paramBlock_sigmoid.set(vec(1.0, 2.0, -0.5, 2.5))
  val dot_sigmoid = Dot(a_sigmoid, paramBlock_sigmoid)
  val sigmoid = Sigmoid(dot_sigmoid)
  println("Dot and Sigmoid ")
  GradientChecker(sigmoid, paramBlock_sigmoid)

  //L2 vector
  val a_regulariz = vec(-1.5, 3.0, 1.5, 0.5)
  val b_regulariz = vec(2.3, 2.0, 1.0, 3.0)
  val paramBlock_regulariz = VectorParam(4)
  paramBlock_regulariz.set(vec(1.0, 2.0, -0.5, 2.5))
  val regulariz = L2Regularization(0.01, a_regulariz, b_regulariz, paramBlock_regulariz)
  println("L2 regularization (vector) ")
  GradientChecker(regulariz, paramBlock_regulariz)


  //L2 Matrix
  /*
  val a_regulariz = vec(-1.5, 3.0, 1.5, 0.5)
  val b_regulariz = vec(2.3, 2.0, 1.0, 3.0)
  val paramBlock_regulariz = VectorParam(4)
  paramBlock_regulariz.set(vec(1.0, 2.0, -0.5, 2.5))
  val regulariz = L2Regularization(0.01, a_regulariz, b_regulariz, paramBlock_regulariz)
  GradientChecker(regulariz, paramBlock_regulariz)
  */

  //Negative log likelihood
  val a_negLog = vec(1.5, 3.0, 1.5, 0.5)
  val paramBlock_negLog = VectorParam(4)
  paramBlock_negLog.set(vec(0.05, 0.01, 0.02, 0.03))
  val dot_negLog = Dot(a_negLog, paramBlock_negLog)
  val negLog_1 = NegativeLogLikelihoodLoss(dot_negLog, 1)
  println("Negative log likelihood with target 1 ")
  GradientChecker(negLog_1, paramBlock_negLog)

  val negLog_0 = NegativeLogLikelihoodLoss(dot_negLog, 0)
  println("Negative log likelihood with target 0 ")
  GradientChecker(negLog_0, paramBlock_negLog)

  //L2 Regularization
  val a_L2Reg = vec(1.5, 3.0, 1.5, 0.5)
  val b_L2Reg = vec(2.5, 1.0, 3.4, 1.5)
  val paramBlock_L2Reg = VectorParam(4)
  paramBlock_L2Reg.set(vec(0.05, 0.01, 0.02, 0.03))
  val l2Model = L2Regularization(10, a_L2Reg, b_L2Reg, paramBlock_L2Reg)
  println("L2 regularization vector")
  GradientChecker(l2Model, paramBlock_L2Reg)

  val a_L2RegMatrix = MatrixParam(3,3)
  val b_L2RegMatrix = MatrixParam(3,3)
  val c_L2RegMatrix = MatrixParam(3,3)
  val l2MatrixModel = L2Regularization(100, a_L2RegMatrix, b_L2RegMatrix, c_L2RegMatrix)
  println("L2 regularization matix")
  GradientChecker(l2MatrixModel, a_L2RegMatrix)
  GradientChecker(l2MatrixModel, b_L2RegMatrix)
  GradientChecker(l2MatrixModel, c_L2RegMatrix)

  //SumOfWordVector scoreSentence
  val wordDim = 10
  val vectorRegularizationStrength = 0.1
  LookupTable.cleanTrainableVector()
  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  val sentenceVector = vec(0.2, 0.4, -0.6, 0.1, 0.2, 0.3, -0.8, 0.1, 0.34, -0.55)
  val sentenceScoreModel = model.scoreSentence(sentenceVector)
  println("SumOfWordVector scoreSentence ")
  GradientChecker(sentenceScoreModel, model.vectorParams("param_w"))

  println("SumOfWordVector wordVectorsToSentenceVector")
  val wordVector1 = vec(0.1, 0.3, -0.7, 0.3, 0.2, 0.3, -0.4, 0.2, 0.22, -0.45)
  val sowvSigmoid = Sigmoid(Dot(wordVector1, model.wordVectorsToSentenceVector(
    Seq(model.wordToVector("Hello"), model.wordToVector("World"),
      model.wordToVector("Neural"), model.wordToVector("Network")))))
  model.vectorParams.values.foreach({
    GradientChecker(sowvSigmoid, _)
  })

  //SumOfWordVector regularizer
  println("SumOfWordVector regularizer ")
  val wordVector2 = vec(0.2, 0.6, -0.2, 0.2, 0.7, 0.4, -0.67, 0.2, 0.33, -0.44)
  val modelParam = VectorParam(10)
  modelParam.set(vec(0.1, 0.23, -0.7, 0.76, 0.2, 0.56, -0.4, 0.34, 0.12, -0.48))
  val regularizer = model.regularizer(Seq(wordVector1, wordVector2, modelParam))
  GradientChecker(regularizer, modelParam)

  println("SumOfWordVector loss ")
  val sentence1 = Seq("Hello", "World", "Neural", "Network")
  val loss = model.loss(sentence1, true)
  model.vectorParams.values.foreach({
    GradientChecker(loss, _)
  })

  //Sum
  println("Sum ")
  val sumA = VectorConstant(vec(1.0,2.0,3.0))
  val sumB = VectorParam(3)
  sumB.set(vec(2.0,3.0,4.0))
  val sumModel = Sigmoid(Dot(sumA, Sum(Seq(sumA, sumB))))
    GradientChecker(sumModel, sumB)

  println("tanh")
  val tanhA = VectorConstant(vec(1.0,2.0,3.0))
  val tanhB = VectorParam(3)
  tanhB.set(vec(2.0,3.0,4.0))
  val tanhModel = Sigmoid(Dot(tanhA, Tanh(tanhB)))
  GradientChecker(tanhModel, tanhB)

  //Mul
  println("Mul")
  val mulVector2 = VectorParam(2)
  mulVector2.set(vec(0.2,0.1))

  val mulMatrix = MatrixParam(2,2)
  mulMatrix.set(mat(2,2) (1.0, 2.0, 3.0, 4.0))
  val mulVector = VectorParam(2)
  mulVector.set(vec(1.0, 2.0))
  val mulModel = Sigmoid(Dot(mulVector2, Mul(mulMatrix, mulVector)))
  //val mulModel = Sigmoid(Dot(mulVector2,  mulVector))
  GradientChecker(mulModel, mulVector2)
  GradientChecker(mulModel, mulMatrix)
  GradientChecker(mulModel, mulVector)

  println("RNN scoreSentence")
  LookupTable.cleanTrainableVector()
  val rnnModel = new RecurrentNeuralNetworkModel(10, 10, 0.1, 0.1)
  val sentenceVectorRNN = vec(0.2, 0.4, -0.6, 0.1, 0.2, 0.3, -0.8, 0.1, 0.34, -0.55)
  val sentenceScoreModelRNN = rnnModel.scoreSentence(sentenceVectorRNN)
  GradientChecker(sentenceScoreModelRNN, rnnModel.vectorParams("param_w"))

  println("RNN wordVectorsToSentenceVector")
  val rnnWordVector1 = VectorParam(10)
  rnnWordVector1.set(vec(0.01, 0.03, -0.07, 0.03, 0.02, 0.03, -0.04, 0.02, 0.022, -0.045))

  val rnnSigmoid = Sigmoid(Dot(rnnWordVector1, rnnModel.wordVectorsToSentenceVector(
    Seq(rnnModel.wordToVector("Hello"), rnnModel.wordToVector("World"),
      rnnModel.wordToVector("Neural"), rnnModel.wordToVector("Network")))))

  rnnModel.vectorParams.values.foreach({
    GradientChecker(rnnSigmoid, _)
  })

  rnnModel.matrixParams.values.foreach({
    GradientChecker(rnnSigmoid, _)
  })

  //RNN regularizer
  println("RNN regularizer")
  val rnnWordVector2 = VectorParam(10)
  rnnWordVector2.set(vec(0.2, 0.6, -0.2, 0.2, 0.7, 0.4, -0.67, 0.2, 0.33, -0.44))
  val rnnModelParam = VectorParam(10)
  rnnModelParam.set(vec(0.1, 0.23, -0.7, 0.76, 0.2, 0.56, -0.4, 0.34, 0.12, -0.48))
  val rnnRegularizer = rnnModel.regularizer(Seq(rnnWordVector1, rnnWordVector2, rnnModelParam))

  GradientChecker(rnnRegularizer, rnnWordVector1)
  GradientChecker(rnnRegularizer, rnnWordVector2)
  GradientChecker(rnnRegularizer, rnnModelParam)
  GradientChecker(rnnRegularizer, rnnModel.vectorParams("param_w"))

  println("RNN loss ")
  val rnnSentence1 = Seq("Hello", "World", "Neural", "Network")
  val rnnLoss = rnnModel.loss(rnnSentence1, true)
  rnnModel.vectorParams.values.foreach({
    GradientChecker(rnnLoss, _)
  })

  rnnModel.matrixParams.values.foreach({
    GradientChecker(rnnLoss, _)
  })

  println("VectoMul")
  val vectorMulA = VectorParam(3)
  vectorMulA.set(vec(0.1, 0.2, 0.3))

  val vectorMulB = VectorParam(3)
  vectorMulB.set(vec(1.0, 2.0, 3.0))

  val vectorMulC = VectorParam(3)
  vectorMulC.set(vec(0.2, 0.3, 0.4))

  val vectorMulSigmoid = Sigmoid(Dot(vectorMulA, VectorMul(vectorMulB, vectorMulC)))
  GradientChecker(vectorMulSigmoid, vectorMulA)
  GradientChecker(vectorMulSigmoid, vectorMulB)
  GradientChecker(vectorMulSigmoid, vectorMulC)


  println("SumMulWordVectorModel")
  LookupTable.cleanTrainableVector()
  val sumMulModel = new SumMultOfWordVectorsModel(3, 0)
  val sumMulA = VectorParam(3)
  sumMulA.set(vec(0.1, 0.2, 0.3))

  val sumMulSentence = Seq("Hello", "World", "DL")
  val sumMulSentenceVector =
    sumMulModel.wordVectorsToSentenceVector(sumMulSentence.map(sumMulModel.wordToVector(_)))
  val sumMulSigmoid = Sigmoid(Dot(sumMulA, sumMulSentenceVector))

  sumMulModel.vectorParams.keySet.foreach(word => {
    GradientChecker(sumMulSigmoid, sumMulModel.vectorParams(word))
  })
  GradientChecker(sumMulSigmoid, sumMulA)

  println("Dropout")
  LookupTable.cleanTrainableVector()
  val dropOutWords = Seq("Hello", "World", "Dropout")
  val dropOutModel = new RecurrentNeuralNetworkModelWithDropout(3, 3, 0, 0, 0.2)
  dropOutModel.isTestTime = false
  val dropoutSentenceVector =
    dropOutModel.wordVectorsToSentenceVector(dropOutWords.map(dropOutModel.wordToVector(_)))

  val sigmoidDropOut = Sigmoid(Dot(dropoutSentenceVector, vec(.3, .4, .6)))

  dropOutModel.vectorParams.keySet.foreach(key => {
    GradientChecker(sigmoidDropOut, dropOutModel.vectorParams(key))
  })

  dropOutModel.isTestTime = true

  dropOutModel.vectorParams.keySet.foreach(key => {
    GradientChecker(sigmoidDropOut, dropOutModel.vectorParams(key))
  })

}
