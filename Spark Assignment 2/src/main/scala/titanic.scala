import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkConf



object titanic {
  def modify(data: DataFrame): DataFrame = {

    val meanValueAge = data.select(mean("Age")).first()(0).asInstanceOf[Double]
    val meanValueFare = data.select(mean("Fare")).first()(0).asInstanceOf[Double]

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

    val modify_data = assembler.transform(dfModified)

    modify_data

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

    val train = modify(traindata).select("features", "Survived")
    val test = modify(testfeatures).select("features")

    val dt = new DecisionTreeClassifier().setLabelCol("Survived").setFeaturesCol("features")

    val dtModel = dt.fit(train)

    val dtPredictions = dtModel.transform(test)

    val dtcsv = dtPredictions.withColumn("PassengerId", monotonically_increasing_id() + 892)

    val dtCSV = dtcsv.withColumnRenamed("prediction", "Survived").select("PassengerId","Survived")

    dtCSV.write.option("header", "true").csv("/Users/junlongqiao/data/dt")

  }
}
