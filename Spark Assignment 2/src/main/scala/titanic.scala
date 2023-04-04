import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkConf


object titanic {
  def modify(data: DataFrame): DataFrame = {
    val meanValueAge = data.select(mean("Age")).first()(0).asInstanceOf[Double]
    val meanValueFare = data.select(mean("Age")).first()(0).asInstanceOf[Double]
    val dfFilled0 = data.na.fill(meanValueAge, Seq("Age"))
    val dfFilled = dfFilled0.na.fill(meanValueFare, Seq("Fare"))
    val dfModified = dfFilled
      .withColumn("Sex", when(col("Sex") === "female", 1).otherwise(0))
      .withColumn("Embarked1", when(col("Embarked") === "S", 1).otherwise(0))
      .withColumn("Embarked2", when(col("Embarked") === "C", 1).otherwise(0))
      .withColumn("Embarked3", when(col("Embarked") === "Q", 1).otherwise(0))


    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked1", "Embarked2", "Embarked3"))
      .setOutputCol("features")
    val idk = assembler.transform(dfModified)
    val preparedData = idk.select("features", "Survived")
    preparedData.show()
    preparedData
  }
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("MyApp")
      .setMaster("local[*]")
      .set("spark.executor.memory", "16g")

    val spark = SparkSession.builder()
      .config(conf)
      .appName("Example")
      .master("local[*]")
      .getOrCreate()

    val traindata = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/junlongqiao/data/train.csv")
      .limit(892)

    val testfeatures = spark
      .read.option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/junlongqiao/data/test.csv")
      .limit(419)

    val testlabel = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/junlongqiao/data/gender_submission.csv")
      .limit(419)

    val testdata = testfeatures.join(testlabel,"PassengerId")

    val train = modify(traindata)
    val test = modify(testdata)

    val dt = new DecisionTreeClassifier().setLabelCol("Survived").setFeaturesCol("features")
    val rf = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features")
    val gbt = new GBTClassifier().setLabelCol("Survived").setFeaturesCol("features")

    val dtModel = dt.fit(train)
    val rfModel = rf.fit(train)
    val gbtModel = gbt.fit(train)

    val dtPredictions = dtModel.transform(test)
    val rfPredictions = rfModel.transform(test)
    val gbtPredictions = gbtModel.transform(test)

    val dtA = dtPredictions.withColumn("Accuracy", when(col("Survived") === col("prediction"), 1).otherwise(0))
    val rfA = rfPredictions.withColumn("Accuracy", when(col("Survived") === col("prediction"), 1).otherwise(0))
    val gbtA = gbtPredictions.withColumn("Accuracy", when(col("Survived") === col("prediction"), 1).otherwise(0))

    val dtAccuracy = dtA.select(mean("Accuracy")).first.getDouble(0)*100
    val rfAccuracy = rfA.select(mean("Accuracy")).first.getDouble(0)*100
    val gbtAccuracy = gbtA.select(mean("Accuracy")).first.getDouble(0)*100

    println(f"$dtAccuracy%.2f%%")
    println(f"$rfAccuracy%.2f%%")
    println(f"$gbtAccuracy%.2f%%")

  }
}
