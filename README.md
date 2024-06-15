# Out-Of-Distribution Is Not Magic: The Clash Between Rejection Rate and Model Success


This repository contains code and results for the research paper titled "Out-Of-Distribution Is Not a Magic: The Clash Between Rejection Rate and Model Success". The goal of this study was to evaluate the performance of different out-of-distribution (OOD) detection techniques in the context of network traffic classification. 

## Abstract
Network traffic classification plays a crucial role in ensuring the security and efficiency of computer networks. However, the presence of unknown or out-of-distribution traffic poses a significant challenge to accurate classification. In this study, we explore the effectiveness of three OOD detection methods - ODIN, GradBP, and K+1 - in enhancing the performance of traffic classification models. We conduct experiments on both binary (benign vs. malicious) and multiclass (well-known applications) classification tasks and evaluate the trade-offs between accuracy and rejection rates. Our findings highlight the task-dependent nature of OOD detection and identify the most suitable method for each classification scenario.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [How to Use](#how-to-use)
- [References](#references)

## Introduction
Network traffic classification is essential for various network management tasks, including intrusion detection, quality of service, and traffic engineering. However, accurately classifying traffic becomes challenging when confronted with unknown or out-of-distribution samples. In this study, we investigate the effectiveness of three OOD detection techniques - ODIN, GradBP, and K+1 - in improving the performance of traffic classification models. We evaluate these methods on both binary and multiclass classification tasks and analyze the trade-offs between accuracy and rejection rates.

## How to Use
Running this code requires the datasets we use, which are not public as of right now.

