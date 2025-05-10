import tensorflow as tf

def soft_update(target, source, tau):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign((1.0 - tau) * target_var + tau * source_var)

def hard_update(target, source):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign(source_var)