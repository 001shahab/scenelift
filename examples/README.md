# Examples

All of them run on a laptop CPU with `pip install -r requirements.txt`.

| File | Shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Reconstruct, list objects, measure, check a fit, find a placement, select by click, edit, save and export. Runs on the synthetic room, or on your footage when the model backends are installed |
| [`llm_tools.py`](llm_tools.py) | The function-calling loop for a language model: tool definitions out, tool calls in, numbers with error bars back |
| [`ground_truth.example.json`](ground_truth.example.json) | The format of tape-measure ground truth read by `scenelift eval` and by the Evaluate tab of SceneLift Studio. Replace the object names and numbers with your own |

```bash
python examples/quickstart.py
python examples/llm_tools.py
```
