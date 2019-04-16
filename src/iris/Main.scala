package iris

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

import scala.util.Random

/**
  * @author JavaEdge
  * @date 2019-04-15
  *
  */
object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("iris")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN") ///日志级别

    val file = spark.read.format("csv").load("iris.data")
    // file.show()

    import spark.implicits._
    val random = new Random()
    val data = file.map(row => {
      // 将列标签替换为数字
      val label = row.getString(4) match {
        case "Iris-setosa" => 0
        case "Iris-versicolor" => 1
        case "Iris-virginica" => 2
      }

      (row.getString(0).toDouble,
        row.getString(1).toDouble,
        row.getString(2).toDouble,
        row.getString(3).toDouble,
        label,
        random.nextDouble())
    }).toDF("_c0", "_c1", "_c2", "_c3", "label", "rand").sort("rand") //.where("label = 1 or label = 0")

    val assembler = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3")).setOutputCol("features")

    val dataset = assembler.transform(data)
    val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2))

    //bayes
    val bayes = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
    val model = bayes.fit(train) //训练数据集进行训练
    model.transform(test).show() //测试数据集进行测试，看看效果如何
  }
}


