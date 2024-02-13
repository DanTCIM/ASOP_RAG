# Actuarial Standards of Practice (ASOP) Q&A machine using Retrieval Augmented Generation (RAG)
## Description
This project aims to create a Retrieval-Augmented Generation (RAG) process for actuaries to ask questions on a set of Actuarial Standards of Practice (ASOP) documents. The RAG process utilizes the power of the Large Language Model (LLM) to provide answers to questions on ASOPs.

However, RAG is not without challenges, i.e., hallucination and inaccuracy. This code allows verifiability by providing the context it used to arrive at those answers. This process enables actuaries to validate the information provided by the LLM, empowering them to make informed decisions. By combining the capabilities of LLM with verifiability, this code offers actuaries a robust tool to leverage LLM technology effectively and extract maximum value.

The current example uses either OpenAI's GPT 3.5 turbo or a local LLM. Using a local LLM can address potential data privacy or security concerns.

Please see a [Jupyter notebook](./notebook/ASOP_RAG.ipynb) for details of the code in the `notebook' folder.

## Example Output

### Question
User prompt: explain ASOP No. 14  
- *Note: Asking this question to LLMs (such as ChatGPT 4) often leads to hallucination as they often explain the ASOP as if it is not repealed or refer to other ASOP documents.*
- *Note: The RAG provides both the answer and context.*

### Answer
ASOP No. 14 was reviewed as part of a project to look at all cash flow testing standards of practice, and relevant portions were incorporated into the revisions of ASOP No. 7 and ASOP No. 22. The ASB voted to adopt the revised ASOP No. 7 and ASOP No. 22 and to repeal ASOP No. 14 in September 2001. The effective date of ASOP No. 7 was deferred to July 15, 2002, while concerns were reviewed regarding its applicability to property/casualty practice.

### Context
*Note: Providing context to the answer allows actuaries to review and validate.*
- Source 1: ../data/ASOP/asop007_128.pdf, page 5:  
virelated to cash flow testing. Finally, the ASB has adopted a new format for standards, and this standard has been rewritten to conform to that new format. In addition to ASOP No. 7, as part of the project to look at all cash flow testing standards of practice, ASOP No. 14 and ASOP No. 22 were al so reviewed. Relevant portions of ASOP No.  
14 were incorporated within the 2001 revisions of ASOP No. 7 and ASOP No. 22. At its September 2001 meeting, the ASB voted to adopt the revised ASOP No. 7 and ASOP No. 22 and to repeal ASOP No. 14. In April 2002, the ASB voted to defer the effective date of ASOP No. 7 to July 15, 2002 while it reviewed concerns raised by the Academy’s Casualty Practice Council regarding the standard’s applicability to property/casualty practice. At its June 2002 meeting, the ASB amended the scope to conform to generally accepted casualty actuarial practice. Please see appendix 3 for further information. Exposure Draft

- Source 2: ../data/ASOP/asop004_173.pdf, page 31:  
are found in the current version of ASOP No. 4. The reviewers believe the reference to Precept 8 remains appropriate. The reviewers do not believe that the proposed change significantly improves the language included in the current version of ASOP No. 4, and made no change.  

- Source 3: ... *(all the context not shown in this illustration)*

## Author
Dan Kim 

- [@LinkedIn](https://www.linkedin.com/in/dan-kim-4aaa4b36/)
- dan.kim.actuary@gmail.com (feel free to reach out with questions or comments)

## Date
- Initially published on 2/12/2024
- The contents may be updated from time to time
  
## License
This project is licensed under the Apache License 2.0- see the LICENSE.md file for details.

## Acknowledgments and References
- https://www.actuarialstandardsboard.org/standards-of-practice/ (downloaded as of December 2023)
- https://python.langchain.com/docs/use_cases/question_answering/quickstart
- https://python.langchain.com/docs/use_cases/question_answering/sources
- https://chat.langchain.com/
