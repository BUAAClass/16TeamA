import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hadoop on 16-5-10.
  */
object Spark {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("meng").setMaster("spark://192.168.1.106:7077").setJars(List("/home/hadoop/IdeaProjects/test2/out/artifacts/test2_jar/test2.jar"))
    val sc = new SparkContext(conf)
    val data=sc.textFile("hdfs://192.168.1.106:9000/user/hadoop/ratings.dat")
    val ratings = data.map(_.split("::") match { case Array(user, item, rate, ts) =>  Rating(user.toInt, item.toInt, rate.toDouble)}).cache()
    val rank = 12
    val lambda = 0.01
    val numIterations = 14
    val model = ALS.train(ratings, rank, numIterations, lambda)
    model.userFeatures
    model.userFeatures.count()
    model.productFeatures
    model.productFeatures.count()

    val usersProducts= ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    var predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }
    predictions.saveAsTextFile("hdfs://192.168.1.106:9000/user/hadoop/output10")
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    ratesAndPreds.count()
    val rmse= math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean())
    def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]) = {
      val usersProducts = data.map { case Rating(user, product, rate) =>
        (user, product)
      }

      val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

      val ratesAndPreds = data.map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.join(predictions)

      math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean())
    }
    sc.stop()
  }
}
