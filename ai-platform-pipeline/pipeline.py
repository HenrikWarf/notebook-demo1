from typing import Text
import absl
import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from aiplatform.pipelines import client

from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental import latest_artifacts_resolver ## demo
from tfx.dsl.experimental import latest_blessed_model_resolver ## demo
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types import channel

PROJECT_ID         = 'crazy-hippo-01'
REGION             = 'us-central1'
API_KEY            = ''
PIPELINE_NAME      = 'earnings_ml_pipeline_01'
PIPELINE_ROOT      = 'gs://crazy-hippo-01/tfx/binary_classification/pipeline'
TRANSFORM_FILE     = 'gs://crazy-hippo-01/tfx/binary_classification/transform.py'
TRAINER_FILE       = 'gs://crazy-hippo-01/tfx/binary_classification/trainer.py'
RAW_DATA = "gs://crazy-hippo-01/tfx/binary_classification/raw/"
SERVING_MODEL_DIR = 'gs://crazy-hippo-01/tfx/binary_classification/serving_model/'



def create_tfx_pipeline(
    pipeline_name: Text, input_dir: Text
):
   
    # Output 2 splits: train:eval=3:1.
    example_gen = CsvExampleGen(input=external_input(RAW_DATA))
    
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)
    
    # Fetch the latest trained model under the same context for warm-starting.
    latest_model_resolver = ResolverNode(
        instance_name='latest_model_resolver',
        resolver_class=latest_artifacts_resolver.LatestArtifactsResolver,
        model=channel.Channel(type=standard_artifacts.Model))
    
    trainer = Trainer(
        module_file=TRAINER_FILE,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=3000),
        eval_args=trainer_pb2.EvalArgs(num_steps=3000))
    
    # Get the latest blessed model for model validation.
    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=channel.Channel(type=standard_artifacts.Model),
        model_blessing=channel.Channel(type=standard_artifacts.ModelBlessing))

    # Set the TFMA config for Model Evaluation and Validation.
    eval_config = tfma.EvalConfig(
       model_specs=[tfma.ModelSpec(label_key='label')],
       slicing_specs=[tfma.SlicingSpec()],
       metrics_specs=[
           tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      # Accept models only if SparseCategoricalAccuracy > 0.8
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.8}),
                      # TODO: modify this
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-2})))
          ])
      ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR)))


    components=[
        example_gen, statistics_gen, schema_gen, example_validator, transform,
        latest_model_resolver, 
        trainer, model_resolver, evaluator, pusher
    ]

    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=PIPELINE_ROOT,
        components=components
    )
    
# Compile and run the pipeline
print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(__import__('tfx.version').__version__))
#absl.logging.set_verbosity(absl.logging.INFO)

tfx_pipeline = create_tfx_pipeline(
        pipeline_name=PIPELINE_NAME,
        input_dir=RAW_DATA
)
client = client.Client(project_id=PROJECT_ID, region=REGION, api_key=API_KEY)


config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
    project_id=PROJECT_ID,
    display_name=PIPELINE_NAME)
runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
    config=config,
    output_filename='pipeline.json')

runner.run(tfx_pipeline, write_out=True)
client.create_run_from_job_spec('pipeline.json')