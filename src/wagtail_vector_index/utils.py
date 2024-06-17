from django.db.models.base import ModelBase


def get_base_concrete_model(model_class: ModelBase) -> ModelBase:
    """
    For a model that uses multi-table-inheritance, this returns the model
    that contains the primary key. For example, for any Wagtail page object,
    this will return the `Page` model.
    """
    parents = model_class._meta.get_parent_list()
    if parents:
        return parents[-1]
    return model_class
