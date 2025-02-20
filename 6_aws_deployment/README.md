# Overview

This project involves deploying a generative language model as a REST API using AWS SageMaker. The model is based on a transformer architecture and is capable of generating text given an input prompt. The deployment is structured to allow inference via a simple HTTP request to an endpoint.

# What was done

## Model preparation

The model and tokenizer were downloaded and packaged into model.tar.gz.

The model directory was structured to be compatible with SageMaker’s inference container.

## Containerized inference service

A Flask-based API was implemented to handle HTTP requests and interact with the model.

The application includes a /ping endpoint for health checks and an /invocations endpoint for text generation.

The model is loaded into memory when the container starts to optimize performance.

## Model inference behavior

The model generates output based on input text, with predefined hyperparameters to control the output.

Certain parameters (e.g., top_k, temperature, and do_sample) were deliberately set to minimal values or disabled, making the generated text highly deterministic and repeatable.

## AWS deployment

The model was deployed on AWS SageMaker using a custom inference container.

The deployment process included building a Docker image, pushing it to AWS Elastic Container Registry (ECR), and setting up a SageMaker endpoint.

# Notes on model behavior

Due to the minimalistic settings, the generated text for the same input will remain almost identical.

The configuration prioritizes stability and deterministic behavior over diversity in generation.
