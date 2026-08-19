---
license: apache-2.0
---

# FlightGPT
The training data and extra data of Flightgpt.

FlightGPT is a state-of-the-art UAV Vision-and-Language Navigation (VLN) framework designed for applications like disaster response, logistics delivery, and urban inspection. Built on powerful Vision-Language Models (VLMs), FlightGPT employs a two-stage training pipeline: supervised fine-tuning (SFT) with high-quality demonstrations to improve initialization and reasoning, followed by Group Relative Policy Optimization (GRPO) guided by a composite reward considering goal accuracy, reasoning quality, and format compliance to enhance generalization. With a Chain-of-Thought (CoT) reasoning mechanism for interpretable decision-making, FlightGPT achieves state-of-the-art performance on the city-scale CityNav dataset, surpassing the strongest baseline by 9.22% in unseen environments.