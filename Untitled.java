---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_125/355914679.py in <cell line: 0>()
    283     from kaggle_secrets import UserSecretsClient
--> 284     os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    285 except Exception as exc:

/usr/local/lib/python3.13/dist-packages/kaggle_secrets.py in get_secret(self, label)
     63         }
---> 64         response_json = self.web_client.make_post_request(request_body, self.GET_USER_SECRET_BY_LABEL_ENDPOINT)
     65         if 'secret' not in response_json:

/usr/local/lib/python3.13/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')

BackendError: Unexpected response from the service. Response: {'errors': ['No user secrets exist for kernel id 139728785 and label HF_TOKEN.'], 'error': {'code': 5}, 'wasSuccessful': False}.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_125/355914679.py in <cell line: 0>()
    285 except Exception as exc:
    286     if TARGET_MODEL_ID.startswith("meta-llama/"):
--> 287         raise RuntimeError("Bật secret HF_TOKEN có quyền đọc model Llama gated") from exc
    288 config = json.loads((CODE_ROOT / "configs" / f"{MODE}.json").read_text())