import apache_beam as beam
from apache_beam.options.pipeline_options import StandardOptions, PipelineOptions

options = PipelineOptions()
options.view_as(StandardOptions).streaming = False


def selectData(element) :
      return element.split(',')

def filtering(record):
      return record[3] == 'Accounts'

#p1 = beam.Pipeline()

pipeline1 = beam.Pipeline(options=options)

#with beam.Pipeline() as pipeline1:
dep_data_count = (
    pipeline1
      |'Read from file' >> beam.io.ReadFromText('gs://crazy-hippo-01/dataflow_beam_data/dept-data.txt')
      |'Select_data' >> beam.Map(selectData)
      |'Filter record on Accounts' >> beam.Filter(filtering)
      |'Create Tuple of Records' >> beam.Map(lambda record : (record[1], 1))
      |'Apply CombinePerKey on Records' >> beam.CombinePerKey(sum)
      |'Write to Cloud Storage' >> beam.io.WriteToText('gs://crazy-hippo-01/dataflow_beam_data/output_new')
)

pipeline1.run().wait_until_finish()