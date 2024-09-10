---
license: gemma
datasets:
- RayBernard/leetcode
language:
- en
base_model: google/gemma-2-2b-it
tags:
- code-generation
- nlp
- transformers
- causal-lm
- coding-education
---
# Model Card for ThinkLink Gemma-2-2B-IT

<!-- Provide a quick summary of what the model is/does. -->

The **ThinkLink Gemma-2-2B-IT** model helps users solve coding test problems by providing guided hints and questions, encouraging self-reflection and critical thinking rather than directly offering solutions.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
This is a fine-tuned version of the **Gemma-2-2B-IT** model, aimed at helping users solve coding problems step by step by providing guided hints and promoting self-reflection. The model does not directly provide solutions but instead asks structured questions to enhance the user's understanding of problem-solving strategies, particularly in coding tests.


- **Developed by:** MinnieMin
<!-- - **Funded by [optional]:** [More Information Needed] -->
<!-- - **Shared by [optional]:** [More Information Needed] -->
- **Model type:** Causal Language Model (AutoModelForCausalLM)
- **Language(s) (NLP):** English (primary)
<!-- - **License:** [More Information Needed] -->
- **Finetuned from model [optional]:** [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)

### Model Sources 

<!-- Provide the basic links for the model. -->

- **Repository:** [gemma-2-2b-it-ThinkLink](https://huggingface.co/MinnieMin/gemma-2-2b-it-ThinkLink)
<!-- - **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed] -->

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be used for educational purposes, especially for coding test preparation. It generates step-by-step problem-solving hints and structured questions to guide users through coding problems.


### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

- This model can be fine-tuned for various types of problem-solving tasks or other domains requiring guided feedback.
- Can be integrated into learning platforms to assist with coding challenges or programming interviews.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Direct code generation without understanding the steps may lead to incorrect or misleading results.
- It is not suitable for tasks that require a detailed and immediate answer to general-purpose questions or advanced mathematical computations.

<!-- ## Bias, Risks, and Limitations

This section is meant to convey both technical and sociotechnical limitations.

[More Information Needed] -->

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be aware that the model encourages self-reflection and thought processes, which may not always align with seeking quick solutions. The model’s effectiveness depends on user interaction and the problem context, and it may not perform well in certain programming domains without further fine-tuning.


## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "MinnieMin/gemma-2-2b-it-ThinkLink"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("What algorithm should I use for this problem?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was fine-tuned on a dataset of structured coding test problems and solutions, focusing on guiding users through the problem-solving process.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

- Training regime: Mixed precision (fp16)
- Hardware: 1 L4 GPU
- Training time: Approximately 17 hours
- Fine-tuning approach: Low-Rank Adaptation (LoRA)


<!-- #### Training Hyperparameters

#### Speeds, Sizes, Times [optional]

This section provides information about throughput, start/end time, checkpoint size if relevant, etc.

[More Information Needed]

## Evaluation

This section describes the evaluation protocols and provides the results.

### Testing Data, Factors & Metrics

#### Testing Data

This should link to a Dataset Card if possible.

[More Information Needed]

#### Factors

These are the things the evaluation is disaggregating by, e.g., subpopulations or domains.

[More Information Needed]

#### Metrics

These are the evaluation metrics being used, ideally with a description of why.

[More Information Needed]

### Results

[More Information Needed] -->

#### Summary
The model was able to effectively guide users through various coding challenges by providing structured hints and questions that promoted deeper understanding.


## Citation 

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@misc{MinnieMin_gemma_2_2b_it_ThinkLink,
  author = {MinnieMin},
  title = {ThinkLink Gemma-2-2B-IT: A Guided Problem-Solving Model},
  year = {2024},
  url = {https://huggingface.co/MinnieMin/gemma-2-2b-it-ThinkLink},
}
