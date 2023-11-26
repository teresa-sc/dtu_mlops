![Logo](../figures/icons/bentoml.png){ align=right width="130"}

# Machine Learning Deployments

---

!!! info "Core Module"

In the [previous module](apis.md) you learned about how to use `FastAPI` to create an API to interact with your machine
learning models. `FastAPI` is a great framework, but it is a general framework meaning that it was not developed with
machine learning applications in mind. This means that there are features which you may consider to be missing when
considering running large scale machine learning models:

1. Micro-batching:

2. Async inference:

3. Native GPU support: you can definitely run part of your application in FastAPI if you want to.

It should come as no surprise that multiple frameworks have therefore sprung up that better supports deployment of
machine learning algorithms:

* [Cortex](https://github.com/cortexlabs/cortex)

* [Bento ML](https://github.com/bentoml/bentoml)

* [Ray Serve](https://docs.ray.io/en/master/serve/)

* [Triton Server](https://github.com/triton-inference-server/server)

* [Torchserve](https://pytorch.org/serve/)

* [Tensorflow serve](https://github.com/tensorflow/serving)

The first 4 frameworks are backend agnostic, meaning that they are intended to work with whatever computational backend
you model is implemented in (Tensorflow vs PyTorch vs Jax), whereas the last two are backend specific to respective
Pytorch and Tensorflow.

It should come as know surprise that there are therefore specific features that
large scale

As you learned about in the previous module,

ASGI (Asynchronous Server Gateway Interface)

## Bento ML core concepts

The three concepts to work

Bento ML support a large set of [frameworks](https://docs.bentoml.com/en/latest/frameworks/index.html)

we are in a [later module](../)

if you are interested in learning how torchserve works, we have an
[optional learning module](../s7_deployment/local_deployment.md) on this framework.

## ❔ Exercises

In general we advice looking through the [docs](https://docs.bentoml.com/en/latest/index.html) for Bento ML if you
need help with any of the exercises.

1. Install BentoML

    ```bash
    pip install bentoml
    ```

2. You are in principal free to serve any Pytorch model you like, but we recommend to start out with the model you
    trained during the [last module on the first day](../s1_development_environment/deep_learning_software.md) for
    recognizing hand written digits. The first step is to save our model in a format that Bento ML can serve. You
    have three options to choose from here:

    === "Pytorch"

        ```python
        import bentoml

        bentoml.pytorch.save_model(
            "my_torch_model",
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        )
        ```

    === "Pytorch + TorchScript"

        The better option is first to compile your model using `torchscript` like this:

        ```python
        scriptet_model = torch.jit.script(model)
        ```

        and then save it with:

        ```
        bentoml.torchscript.save_model
        ```

    === "Pytorch Lightning"

        If you model was trained in Pytorch Lightning e.g. you re-implemented it after following
        [this module](../s4_debugging_and_logging/boilerplate.md) then you can also choose to save the model with

        ```python

        ```

    Take either a pre-trained model you have on your computer, load it in and save it again using the above code.
    Alternatively, you can just retrain your model and add the code above to save the model.

3. Regardless of how you choose to save the model, you can also add additional information when saving the model,
    such ad labels and metadata:

    ```python
    bentoml.pytorch.save_model(
        "demo_mnist",   # Model name in the local Model Store
        trained_model,  # Model instance being saved
        labels={    # User-defined labels for managing models in BentoCloud
            "owner": "nlp_team",
            "stage": "dev",
        },
        metadata={  # User-defined additional metadata
            "acc": acc,
            "cv_stats": cv_stats,
            "dataset_version": "20210820",
        },
    )
    ```

    add at least one labels and one metadata key to your saving call.

4. Bento ML come with its own command line interface which provide an easy way to interact with your saved models. Try
    out the commands:

    ```bash
    bentoml models list
    bentoml models get <model-name>
    bentoml models export <model-name>
    ```

5. In addition
