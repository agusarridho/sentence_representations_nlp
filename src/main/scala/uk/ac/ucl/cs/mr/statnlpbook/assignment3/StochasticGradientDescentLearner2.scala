package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
  * Problem 2
  */
object StochasticGradientDescentLearner2 extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double, Double) => Double): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    //val globalWeightVector = model.vectorParams("param_w")
    var updatedLearningRate = learningRate

    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
        //todo: update the parameters of the model and accumulate the loss

        val loss = model.loss(sentence, target)

        accLoss = accLoss + loss.forward()
        loss.backward()
        loss.update(updatedLearningRate)
      }
      //val a = LookupTable.get("next")
      updatedLearningRate = epochHook(i, accLoss, updatedLearningRate)
    }
  }
}