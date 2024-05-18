<p align="center">
  <img src="https://github.com/Docu-Mint/DocuMint/assets/96804013/a5ae74ce-12a2-495c-9610-8b9e7655b1b7" width="250px" alt="Alt Text">
</p>

# DocuMint: Docstring Generation for Python using Small Language Models ([Paper](https://arxiv.org/abs/2405.10243), [Slides](https://github.com/Docu-Mint/DocuMint/blob/main/Presentation.pdf))

This research investigates the efficacy of Small Language Models (SLMs) for Code in generating high-quality docstrings by assessing accuracy, conciseness, and clarity. We benchmark the performance of leading CodeSLMs in docstring generation quantitatively through mathematical formulas and qualitatively through human evaluation using Likert scale. We also release DocuMint as a large-scale supervised fine-tuning dataset with 100,000 samples. Lastly, make use of the dataset to fine-tune CodeGemma 2B model using LoRA. The dataset and the fine-tuned model can be found in [HuggingFace](https://huggingface.co/documint) 

## Experiment Results

<p align="center">
  <img src="https://github.com/Docu-Mint/DocuMint/assets/96804013/426b8af9-5926-474d-ac17-15a873680bd6" alt="Alt Text">
  <br>
  <i>In quantitative experiments,Llama3 8B achieved the best performance across all metrics, with conciseness and clarity scores of 0.605 and 64.88, respectively. </i>
</p>

<p align="center">
  <img src="https://github.com/Docu-Mint/DocuMint/assets/96804013/7b1e7042-1196-4a05-b225-1f719d13bf26" width="500px" alt="Alt Text">
  <br>
  <i>Under qualitative human evaluation, CodeGemma 7B achieved the highest overall score with an average of 8.3 out of 10 across all metrics.</i>
</p>


## Fine-Tuning using LoRA

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Hyperparameter</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Fine-tuning Method</td>
        <td>LoRA</td>
      </tr>
      <tr>
        <td>Epochs</td>
        <td>4</td>
      </tr>
      <tr>
        <td>Batch Size</td>
        <td>8</td>
      </tr>
      <tr>
        <td>Gradient Accumulation Steps</td>
        <td>16</td>
      </tr>
      <tr>
        <td>Initial Learning Rate</td>
        <td>2e-4</td>
      </tr>
      <tr>
        <td>LoRA Parameters</td>
        <td>78,446,592</td>
      </tr>
      <tr>
        <td>Training Tokens</td>
        <td>185,040,896</td>
      </tr>
    </tbody>
  </table>
</div>


<p align="center">
  Fine-tuning hyperparameters.
</p>

<p align="center">
  <img src="https://github.com/Docu-Mint/DocuMint/assets/96804013/2f768c79-ac17-4253-9402-f1b7e43f1e78" width="500px" alt="Alt Text">
  <br>
  <i>Loss curve during fine-tuning of the CodeGemma 2B base model.</i>
</p>

<p align="center">
  <img src="https://github.com/Docu-Mint/DocuMint/assets/96804013/cf80d585-06e3-4f2c-97f9-d5a1097deaf5" width="500px" alt="Alt Text">
  <br>
  <i>Fine-tuning the CodeGemma 2B model using the DocuMint dataset led to significant improvements in performance across all metrics, with gains of up to 22.5% in conciseness.</i>
</p>


-------
## Cite

```
@article{poudel2024documint,
  title={DocuMint: Docstring Generation for Python using Small Language Models},
  author={Poudel, Bibek and Cook, Adam and Traore, Sekou and Ameli, Shelah},
  journal={arXiv preprint arXiv:2405.10243)},
  year={2024}
}
```

-------
## Acknowledgments

We would like to thank Dr. Audris Mockus for his guidance on the project and help with World of Code. We would also like to thank the Fluidic City Lab for providing the compute resources. 
