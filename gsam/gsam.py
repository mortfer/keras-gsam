import copy
import tensorflow as tf
from keras.layers import deserialize as deserialize_layer
from keras.saving import serialize_keras_object

class GSAM(tf.keras.Model):
    """Surrogate Gap Guided Sharpness-Aware Minimization
    (GSAM) training flow. https://arxiv.org/abs/2203.08065
    """
    def __init__(self, model, rho=0.05, alpha=0.1, eps=1e-12, name=None):
        super().__init__(name=name)
        self.model = model
        self.rho = rho
        self.alpha = alpha
        self.eps = eps
        
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
            
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.compiled_loss(y, predictions)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._gradients_order2_norm(gradients)
        scale = self.rho / (grad_norm + self.eps)
        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            self._distributed_apply_epsilon_w(
                param, e_w, tf.distribute.get_strategy()
            )
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.compiled_loss(y, predictions)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        sam_grad_norm = self._gradients_order2_norm(sam_gradients)
        for (param, e_w) in zip(trainable_params, e_ws):
            # Restore the variable to its original value before
            # `apply_gradients()`.
            self._distributed_apply_epsilon_w(
                    param, -e_w, tf.distribute.get_strategy()
            )

        dot_product = sum(
            [tf.reduce_sum(grad * sam_grad) for grad, sam_grad in zip(gradients,sam_gradients)
                if grad is not None and sam_grad is not None]
        )
        grads_parallel = [sam_grad*dot_product/(sam_grad_norm**2 + self.eps) for sam_grad in sam_gradients]
        grads_orthogonal = [grad-grad_par  for grad,grad_par in zip(gradients,grads_parallel)]

        final_grad = [sam_grad-self.alpha*grad_ortho  for sam_grad,grad_ortho in zip(sam_gradients,grads_orthogonal)]

        self.optimizer.apply_gradients(
            zip(final_grad, trainable_params))
         
        self.compiled_metrics.update_state(y, predictions, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs):
        """Forward pass of GSAM.
        GSAM delegates the forward pass call to the wrapped model.
        Args:
          inputs: Tensor. The model inputs.
        Returns:
          A Tensor, the outputs of the wrapped model for given `inputs`.
        """
        return self.model(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": serialize_keras_object(self.model),
                "rho": self.rho,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Avoid mutating the input dict.
        config = copy.deepcopy(config)
        model = deserialize_layer(
            config.pop("model"), custom_objects=custom_objects
        )
        config["model"] = model
        return super().from_config(config, custom_objects)
    
    def _distributed_apply_epsilon_w(self, var, epsilon_w, strategy):
        # Helper function to apply epsilon_w on model variables.
        if isinstance(
            tf.distribute.get_strategy(),
            (
                tf.distribute.experimental.ParameterServerStrategy,
                tf.distribute.experimental.CentralStorageStrategy,
            ),
        ):
            # Under PSS and CSS, the AggregatingVariable has to be kept in sync.
            def distribute_apply(strategy, var, epsilon_w):
                strategy.extended.update(
                    var,
                    lambda x, y: x.assign_add(y),
                    args=(epsilon_w,),
                    group=False,
                )

            tf.__internal__.distribute.interim.maybe_merge_call(
                distribute_apply, tf.distribute.get_strategy(), var, epsilon_w
            )
        else:
            var.assign_add(epsilon_w)
            
    def _gradients_order2_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm