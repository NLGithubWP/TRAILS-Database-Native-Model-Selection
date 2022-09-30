- NN Successfully,
  - Model selection definition: it refers to selecting a good-performance model giving a specific prediction task. 

- Manually design (detail) 

- NAS (detail): Search space &. Search strategy & Architecture evaluation

- Training-based 

  - TB Architecture evaluation (formulate, definition) ( NASNET, one-shot, predictor training problem)

- Training-free NAS

  - TF Architecture evaluation (formulate, definition)
  - It’s advantages. 3x speed

- Use case & Motivating example.

  - Importance of Model Selection
  - An average user cannot afford high-end hardware and expensive computing service => So it requires an affordable model selection system.
    - Afford definition: It is to achieve model selection in a fast and resource-efficient manner,
  - A user typically has a time budget => and requires model selection anytime performance (definition)
    - Propose a new term, Anytime Model Selection. Definition: It is to select a good-performing model within a given time budge T.

- challenge/gap/limitation:

  - Affordable Model selection challenge/gap/limitation:

    - Training-based architecture evaluation requires 100-1k of full model training, and each model training requires thousands of iterations on high-end GPU. 
    - first - it evaluates one arch very fast. 
    - second - it cannot achieve a resource-efficient manner.

  - Anytime Model Selection challenge/gap/limitation: 
    - evaluating one arch T_a is the minimum execution unit-> evaluating one arch for Training-based arch evaluation is expensive
    - first - it explores fewer arch within T
    - second/further - less data (a, s) to fit the search strategy, and thus cannot find good-performing arches
    - The Key challenge is how to explore more architecture given the time budget to fit the search strategy better.

  - The existing system cannot support two features.

- The current system  
   
  - Affordable 
  - Anytime performance

- Objective => We present a end-2-end extremely fast model selection system 

  - Affordable MS

    - trining-free arch eval can evaluate one arch with only one forward/backward using a mini-batch 
    - it can evaluate very fast. 
    - it can execute on either GPU or CPU
    - => it is fast and resource-efficiency.

  - Anytime MS

    - training-free arch eval can explore more arches.

- Contribution

  - We present an end-2-end extremely fast model selection system for end users with Affordable and Anytime MS.
  - Benchmarked arch evaluation metrics and proposed a learning-based combination method.
  - System features.
  - Conduct experiments.

