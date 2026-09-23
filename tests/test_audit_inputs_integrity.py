"""Fictional file-based input contracts; temporary files are cleaned after tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from agents.audit_inputs import load_bundle, digest, configuration_digest, verify_adapter_provenance


class AuditInputTests(unittest.TestCase):
    def bundle(self, root):
        manifest=dict(schema="eafmas.audit-inputs.v1",synthetic=True,model_identity="fictional-model",
                      source_revision="fictional-revision",split_definition={},files={})
        for day, split in enumerate(("train","val","test"),1):
            time=f"2040-01-0{day}T00:00:00"
            manifest["split_definition"][split]={"start":time,"end":f"2040-01-0{day+1}T00:00:00"}
            row=dict(request_id=split,split=split,model_identity="fictional-model",forecast_origin=time,timestamps=[time])
            path=root/(split+".jsonl")
            path.write_text(json.dumps(row)+"\n")
            manifest["files"][split]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),count=1)
        path=root/"memory.jsonl"; path.write_text("")
        manifest["files"]["residual_memory"]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),count=0)
        return manifest

    def load(self, root, manifest, **kwargs):
        path=root/"manifest.json"; path.write_text(json.dumps(manifest))
        return load_bundle(path,"test",**kwargs)

    def test_formal_run_refuses_synthetic_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=self.bundle(root)
            with self.assertRaisesRegex(ValueError,"real artifacts"): self.load(root,manifest)
            self.assertEqual(len(self.load(root,manifest,allow_synthetic=True)[1]),1)

    def test_disjoint_splits_and_no_duplicate_origins(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=self.bundle(root)
            bad=deepcopy(manifest)
            bad["files"]["val"]=bad["files"]["train"]
            with self.assertRaises(ValueError): self.load(root,bad,allow_synthetic=True)
            bad=deepcopy(manifest)
            bad["split_definition"]["val"]["start"]="2039-01-01T00:00:00"
            with self.assertRaisesRegex(ValueError,"overlap"): self.load(root,bad,allow_synthetic=True)

    def test_changed_file_and_population_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=self.bundle(root)
            bad=deepcopy(manifest); bad["files"]["test"]["count"]=2
            with self.assertRaisesRegex(ValueError,"Population"): self.load(root,bad,allow_synthetic=True)
            (root/"test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"hash"): self.load(root,manifest,allow_synthetic=True)

    def test_adapter_config_and_population_must_match(self):
        manifest={"model_identity":"fictional"}; config={"rho_full":.03,"device":"cpu"}
        adapter=dict(model_identity="fictional",training_manifest=dict(input_manifest_digest=digest(manifest),configuration_digest=configuration_digest(config)))
        verify_adapter_provenance(adapter,manifest,config)
        verify_adapter_provenance(adapter,manifest,dict(config,device="cuda"))
        with self.assertRaises(ValueError): verify_adapter_provenance(adapter,manifest,dict(config,rho_full=.02))
        with self.assertRaises(ValueError): verify_adapter_provenance(adapter,dict(manifest,changed=True),config)


if __name__=="__main__": unittest.main()
