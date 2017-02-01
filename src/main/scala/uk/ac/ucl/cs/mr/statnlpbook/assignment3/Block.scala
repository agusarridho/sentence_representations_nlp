package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{DenseMatrix => Matrix, DenseVector => Vector}
import breeze.numerics.{log, tanh}

/**
 * @author rockt
 */

/**
 * A trait for the core building **block** of our computation graphs
  *
  * @tparam T the type parameter this block evaluates to (can be Double, Vector, Matrix)
 */
trait Block[T] {
  //caches output after call of forward
  var output: T = _
  //fun fact: when you say "forward" or "back" your lips move in the respective direction
  def forward(): T
  //assumes that forward has been called first!
  def backward(gradient: T): Unit
  //updates parameters of the block
  def update(learningRate: Double)
}

/**
 * A loss function is a block that evaluates to a single double.
 * Generally loss functions don't have upstream gradients,
 * so we can provide an implementation of backward without arguments.
 */
trait Loss extends Block[Double] {
  def backward(): Unit
}

trait ParamBlock[P] extends Block[P] {
  var param: P
  val gradParam: P
  def initialize(dist: () => Double): P
  def set(p: P): Unit = {
    param = p
  }
  def resetGradient(): Unit
}

trait DefaultInitialization {
  def defaultInitialization(): Double
}

/**
 * This trait defines a default way of initializing weights of parameters
 */
trait GaussianDefaultInitialization extends DefaultInitialization {
  def defaultInitialization(): Double = random.nextGaussian() * 0.1
}

/**
 * A simple block that represents a constant double value
  *
  * @param arg the constant double value
 */
case class DoubleConstant(arg: Double) extends Block[Double] with Loss {
  output = arg
  def forward(): Double = output
  def backward(gradient: Double): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
  def backward(): Unit = {} //nothing to do since fixed
}

/**
 * A simple block that represents a constant vector
  *
  * @param arg the constant vector
 */
case class VectorConstant(arg: Vector) extends Block[Vector] {
  output = arg
  def forward(): Vector = output
  def backward(gradient: Vector): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
}

/**
 * A block representing a sum of doubles
  *
  * @param args a sequence of blocks that evaluate to doubles
 */
case class DoubleSum(args: Block[Double]*) extends Block[Double] {
  def forward(): Double = {
    output = args.map(_.forward()).sum
    output
  }
  def backward(gradient: Double): Unit = args.foreach(_.backward(gradient))
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
}

class LossSum(override val args: Loss*) extends DoubleSum(args:_*) with Loss {
  def backward(): Unit = args.foreach(_.backward())
}

/**
 * Problem 2
 */

/**
 * A block representing a vector parameter
  *
  * @param dim dimension of the vector
 * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
 */
case class VectorParam(dim: Int, clip: Double = 10.0) extends ParamBlock[Vector] with GaussianDefaultInitialization {
  var param: Vector = initialize(defaultInitialization) //todo: initialize using default initialization
  val gradParam: Vector = Vector.zeros[Double](dim) //todo: initialize with zeros
  /**
   * @return the current value of the vector parameter and caches it into output
   */
  def forward(): Vector = {
    output = param
    output
  }
  /**
   * Accumulates the gradient in gradParam
    *
    * @param gradient an upstream gradient
   */
  def backward(gradient: Vector): Unit = {
    gradParam :+= gradient //In-place
  }
  /**
   * Resets gradParam to zero
   */
  def resetGradient(): Unit = {
    gradParam := 0.0
  }
  /**
   * Updates param using the accumulated gradient. Clips the gradient to the interval (-clip, clip) before the update
    *
    * @param learningRate learning rate used for the update
   */
  def update(learningRate: Double): Unit = {
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate) //in-place
    resetGradient()
  }
  /**
   * Initializes the parameter randomly using a sampling function
    *
    * @param dist sampling function
   * @return the random parameter vector
   */
  def initialize(dist: () => Double): Vector = {
    param = randVec(dim, dist)
    param
  }
}

/**
 * A block representing the sum of vectors
  *
  * @param args a sequence of blocks that evaluate to vectors
 */
case class Sum(args: Seq[Block[Vector]]) extends Block[Vector] {
  def forward(): Vector = {

    //args.foreach(a => output :+= a.forward()) // element-wise summation of vectors representing individual words

    if (args.length == 1){
      output = args(0).forward().copy
    }else if (args.length > 1){
      output = args(0).forward().copy
      for(i <- 1 to args.length - 1){
        output :+= args(i).forward()
      }
    }
    //output = args.map(_.forward).sum
    output
  }
  def backward(gradient: Vector): Unit = {
    args.foreach(_.backward(gradient))
  }
  def update(learningRate: Double): Unit = {
    args.foreach(_.update(learningRate))
  }
}

/**
 * A block representing the dot product between two vectors
  *
  * @param arg1 left block that evaluates to a vector
 * @param arg2 right block that evaluates to a vector
 */
case class Dot(arg1: Block[Vector], arg2: Block[Vector]) extends Block[Double] {
  def forward(): Double = {
    output = arg1.forward() dot arg2.forward()
    output
  }
  def backward(gradient: Double): Unit = {
    arg1.backward(arg2.output * gradient) // d(xy)/dx = y
    arg2.backward(arg1.output * gradient) // d(xy)/dy = x
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
 * A block representing the sigmoid of a scalar value
  *
  * @param arg a block that evaluates to a double
 */
case class Sigmoid(arg: Block[Double]) extends Block[Double] {
  def forward(): Double = {
    output = breeze.numerics.sigmoid(arg.forward())
    output
  }
  def backward(gradient: Double): Unit = {
    arg.backward(gradient * output * (1 - output))
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
 * A block representing the negative log-likelihood loss
  *
  * @param arg a block evaluating to a scalar value
 * @param target the target value (1.0 positive sentiment, 0.0 negative sentiment)
 */
case class NegativeLogLikelihoodLoss(arg: Block[Double], target: Double) extends Loss {
  //The cost function
  def forward(): Double = {
    val f_theta = arg.forward()
    output = -target * log(f_theta) - (1 - target) * (log(1 - f_theta))
    output
  }
  //loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  //Gradient of the cost function. We want to minimize the cost
  def backward(): Unit = {
    val targetOne_part = -target / arg.output
    val targetZero_part = (1 - target) / (arg.output - 1)
    arg.backward((targetOne_part - targetZero_part))
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
 * A block representing the l2 regularization of a vector or matrix
  *
  * @param strength the strength of the regularization (often denoted as lambda)
 * @param args a block evaluating to a vector or matrix
 * @tparam P type of the input block (we assume this is Block[Vector] or Block[Matrix]
 */
case class L2Regularization[P](strength: Double, args: Block[P]*) extends Loss {
  def forward(): Double = {
    /**
     * Calculates the loss individually for every vector/matrix parameter in args
     */
    val losses = args.map(arg => {
      val in = arg.forward()
      in match {
        case v: Vector => v dot v * (strength * 0.5)
        case w: Matrix => {
          val v = w.toDenseVector
          v dot v * (strength * 0.5)
        }
      }
    })
    output = losses.sum //sums the losses up
    output
  }
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
  //loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  def backward(): Unit = args.foreach(x => x.backward((x.forward() match {
    case v: Vector => v * strength
    case w: Matrix => w * strength
  }).asInstanceOf[P]))
}



/**
 * Problem 3
 */

/**
 * A block representing a matrix parameter
  *
  * @param dim1 first dimension of the matrix
 * @param dim2 second dimension of the matrix
 * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
 */
case class MatrixParam(dim1: Int, dim2: Int, clip: Double = 10.0) extends ParamBlock[Matrix] with GaussianDefaultInitialization {
  var param: Matrix = randMat(dim1, dim2, defaultInitialization)
  val gradParam: Matrix = Matrix.zeros[Double](dim1, dim2)
  def forward(): Matrix = {
    output = param
    param
  }
  def backward(gradient: Matrix): Unit = {
    gradParam :+= gradient
  }
  def resetGradient(): Unit = {
    gradParam := 0.0
  }
  def update(learningRate: Double): Unit = {
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate) //in-place
    resetGradient()
  }
  def initialize(dist: () => Double): Matrix = {
    param = randMat(dim1, dim2, dist)
    param
  }
}

/**
 * A block representing matrix-vector multiplication
  *
  * @param arg1 the left block evaluating to a matrix
 * @param arg2 the right block evaluation to a vector
 */
case class Mul(arg1: Block[Matrix], arg2: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = arg1.forward() * arg2.forward()
    output
  }
  def backward(gradient: Vector): Unit = {
    arg1.backward(outer(gradient, arg2.output))
    arg2.backward(arg1.output.t * gradient)
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
 * A block rerpesenting the element-wise application of the tanh function to a vector
  *
  * @param arg a block evaluating to a vector
 */
case class Tanh(arg: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = tanh(arg.forward())
    output
  }
  def backward(gradient: Vector): Unit = {
    var v: Vector = output :* output
    v = -v + 1.0
    arg.backward(v :* gradient) //Have to check compilation error
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
  * A block representing elementwise vecto-vector multiplication
  *
  * @param arg1 the left block evaluating to a matrix
  * @param arg2 the right block evaluation to a vector
  */
case class VectorMul(arg1: Block[Vector], arg2: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = arg1.forward() :* arg2.forward()
    output
  }
  def backward(gradient: Vector): Unit = {
    arg1.backward(gradient :* arg2.output)
    arg2.backward(gradient :* arg1.output)
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
 * Problem 4
 */

/**
 * A potentially useful block for training a better model (https://en.wikipedia.org/wiki/Dropout_(neural_networks))
  *
  * @param dropoutProb dropout probability
 * @param arg a block evaluating to a vector whose components we want to drop
 */
case class Dropout(dropoutProb: Double, arg: Block[Vector], isTestTime: Boolean) extends Block[Vector] {
  var massVector = vec(1)
  var recalculateMassVector = true
  def forward(): Vector = {
    if (isTestTime)
      output = arg.forward() * (1 - dropoutProb)
    else {
      if (recalculateMassVector) {
        massVector = getMass()
        recalculateMassVector = false
      }

      output = arg.forward() :* massVector
    }
    output
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
  def backward(gradient: Vector): Unit = {
    if (isTestTime){
      //println(gradient)
      arg.backward(gradient * (1 - dropoutProb))

    }else{
      //val v = output :* gradient
      arg.backward(gradient :* massVector)
    }

  }
  def getMass(): Vector = {
    val massVector = Vector.fill[Double](arg.forward().activeSize) {
        if (random.nextDouble <= dropoutProb) 0 else 1
    }
    massVector
  }
}



/**
 * ... be free, be creative :)
 */