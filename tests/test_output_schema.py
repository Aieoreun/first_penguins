import unittest
from pydantic import ValidationError
from guard_review import OutputReviewReport,review_json_schema
from test_observation_review import observation

class OutputSchemaTests(unittest.TestCase):
 def test_observations_have_no_final_values(self):
  self.assertNotIn('grounded',observation().model_dump()['grounding'][0])
 def test_safety_has_no_grounding_fields(self):
  d=observation().model_dump();d['safety']['grounded']=True
  with self.assertRaises(ValidationError):OutputReviewReport.model_validate(d)
 def test_unknown_relation_rejected(self):
  d=observation().model_dump();d['grounding'][0]['relation']='revise'
  with self.assertRaises(ValidationError):OutputReviewReport.model_validate(d)
 def test_safety_cannot_research(self):
  d=observation().model_dump();d['safety']['relation']='insufficient'
  with self.assertRaises(ValidationError):OutputReviewReport.model_validate(d)
 def test_schema_is_output_only(self):
  s=review_json_schema(targets=('grounding','safety'))
  self.assertEqual(set(s['properties']),{'grounding','safety'})
  self.assertNotIn('InputVerdict',s['$defs'])
 def test_missing_candidate_span_rejected(self):
  d=observation().model_dump();del d['grounding'][0]['candidate_span']
  with self.assertRaises(ValidationError):OutputReviewReport.model_validate(d)
