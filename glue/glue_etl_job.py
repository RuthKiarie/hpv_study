"""
AWS Glue ETL job: raw zone -> feature zone.
--RAW_PATH      s3://rwk-hpv-mlops/raw/kenya_hpv_screening/dt=<date>/kenya_hpv_screening_synthetic.csv
  --FEATURE_PATH  s3://rwk-hpv-mlops/features/kenya_hpv_screening/
  --DT            2026-09-19   (partition value written into the output)
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
 
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'RAW_PATH', 'FEATURE_PATH', 'DT'])
 
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
 
raw_path = args['RAW_PATH']
feature_path = args['FEATURE_PATH']
dt = args['DT']
 
# ---- read raw CSV ----
df = spark.read.option("header", "true").option("inferSchema", "true").csv(raw_path)
 
# ---- ordinal encodings ----
edu_map = F.create_map(
    F.lit("none"), F.lit(0),
    F.lit("primary"), F.lit(1),
    F.lit("secondary"), F.lit(2),
    F.lit("higher"), F.lit(3),
)
df = df.withColumn("education_ord", edu_map[F.col("education")])
 
wealth_map = F.create_map(
    F.lit("poorest"), F.lit(0),
    F.lit("poorer"), F.lit(1),
    F.lit("middle"), F.lit(2),
    F.lit("richer"), F.lit(3),
    F.lit("richest"), F.lit(4),
)
df = df.withColumn("wealth_ord", wealth_map[F.col("wealth_quintile")])
 
# ---- binary encodings ----
df = df.withColumn("is_urban", F.when(F.col("residence") == "urban", 1).otherwise(0))
df = df.withColumn("hiv_positive", F.when(F.col("hiv_status") == "positive", 1).otherwise(0))
df = df.withColumn("hiv_unknown", F.when(F.col("hiv_status") == "unknown", 1).otherwise(0))
df = df.withColumn("distance_barrier", F.when(F.col("distance_problem") == "big_problem", 1).otherwise(0))
 
# ---- age bucketing ----
df = df.withColumn(
    "age_group",
    F.when(F.col("age") < 20, "15-19")
     .when(F.col("age") < 30, "20-29")
     .when(F.col("age") < 40, "30-39")
     .otherwise("40-49")
)
 
# ---- parity capping ----
df = df.withColumn(
    "parity_capped",
    F.when(F.col("parity") > 6, 6).otherwise(F.col("parity"))
)
 
# ---- deliberate risk feature: HIV+ AND not yet screened ----
df = df.withColumn(
    "high_risk_flag",
    F.when(
        (F.col("hiv_positive") == 1) & (F.col("screened_last_3yrs") == 0), 1
    ).otherwise(0)
)
 
# ---- partition column for the feature zone ----
df = df.withColumn("dt", F.lit(dt))
 
# ---- write partitioned Parquet to the feature zone ----
df.write.mode("overwrite").partitionBy("dt").parquet(feature_path)
 
job.commit()
 