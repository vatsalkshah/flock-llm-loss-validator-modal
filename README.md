## Implementation of [FLock's LLM Loss Validator](https://github.com/FLock-io/llm-loss-validator) script on modal

### Steps to run it

1. `git clone https://github.com/vatsalkshah/flock-llm-loss-validator-modal.git`
2. `pip install -r requirements.txt`
3. Add FLOCK_API_KEY, HF_TOKEN, TASK_ID in `src/modal_start.py`. You can also select GPU based on the task
4. `modal create volume flock-validator-models`
5. `modal deploy src/modal_start.py`

### Benefits of running it on modal

Modal spins up a GPU container only when there is a model to validate thus 
- saving 75% on GPU costs during the submission phase
- ensuring that one bad model will not stop the validation

### To-dos

- [ ] remove additional dependencies from CPU server for faster boot times
- [ ] ensure that only one instance of an assignment is running at a time
