#!/bin/bash

# Configuration du cluster EMR pour Fruits Classification
# Ce script crée un cluster EMR avec protection de terminaison

set -e

# ============================================
# CONFIGURATION (à adapter selon vos besoins)
# ============================================

CLUSTER_NAME="p9-fruits-classification-cluster"
REGION="eu-west-3"  # Paris (RGPD)
S3_BUCKET="s3://p9-fruits-data"
KEY_NAME="votre-cle-ssh"  # ← À MODIFIER
EMR_RELEASE="emr-6.15.0"

# Types d'instances
MASTER_INSTANCE="m5.xlarge"
CORE_INSTANCE="m5.xlarge"
CORE_INSTANCE_COUNT=1  # 1 instance Core = 1 instance Task selon AWS

# ============================================
# CRÉATION DU CLUSTER
# ============================================

echo "=========================================="
echo "Création du cluster EMR"
echo "Nom: $CLUSTER_NAME"
echo "Région: $REGION"
echo "=========================================="

CLUSTER_ID=$(aws emr create-cluster \
  --name "$CLUSTER_NAME" \
  --region "$REGION" \
  --release-label "$EMR_RELEASE" \
  --applications Name=Spark Name=Hadoop Name=JupyterHub \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceType="$MASTER_INSTANCE",InstanceCount=1 \
    InstanceGroupType=CORE,InstanceType="$CORE_INSTANCE",InstanceCount="$CORE_INSTANCE_COUNT" \
  --bootstrap-actions Path="$S3_BUCKET/scripts/bootstrap_emr.sh" \
  --ec2-attributes KeyName="$KEY_NAME" \
  --use-default-roles \
  --termination-protected \
  --log-uri "$S3_BUCKET/logs/" \
  --enable-debugging \
  --configurations '[
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.executor.memory": "4g",
        "spark.driver.memory": "4g",
        "spark.executor.cores": "2",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "256",
        "spark.sql.parquet.writeLegacyFormat": "true"
      }
    },
    {
      "Classification": "spark-env",
      "Configurations": [
        {
          "Classification": "export",
          "Properties": {
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "SPARK_LOCAL_IP": "127.0.0.1"
          }
        }
      ]
    }
  ]' \
  --query 'ClusterId' \
  --output text)

echo "=========================================="
echo "✓ Cluster créé avec succès"
echo "Cluster ID: $CLUSTER_ID"
echo "=========================================="
echo ""
echo "Commandes utiles :"
echo "  - Vérifier le statut:"
echo "    aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION --query 'Cluster.Status.State'"
echo ""
echo "  - Obtenir le DNS du Master:"
echo "    aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION --query 'Cluster.MasterPublicDnsName'"
echo ""
echo "  - Supprimer la protection puis terminer:"
echo "    aws emr modify-cluster-attributes --cluster-id $CLUSTER_ID --region $REGION --no-termination-protected"
echo "    aws emr terminate-clusters --cluster-ids $CLUSTER_ID --region $REGION"
echo "=========================================="

# Attendre que le cluster soit prêt (optionnel)
echo ""
echo "Attente du démarrage du cluster..."
aws emr wait cluster-running --cluster-id "$CLUSTER_ID" --region "$REGION"

echo "✓ Cluster en cours d'exécution (état: WAITING)"
echo ""
echo "DNS du Master:"
aws emr describe-cluster --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Cluster.MasterPublicDnsName' --output text
