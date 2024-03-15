# Architecture

This page describes the low-level architecture of `wagtail-vector-index`. It is intended for those who wish to contribute to the package, or to customise it's behaviour at the lowest levels.

The APIs described here should not be considered 'public' or final unless they are documented elsewhere.

The main goal of `wagtail-vector-index` is to allow a developer to generate and store vector embeddings of 'things' (mainly Wagtail Pages). To do this, and to allow some flexibility in how that is done, that process is broken down in to a few key components:

## `VectorIndex` - `index/base.py`

`VectorIndex`s are the type most developers will be interacting with. A `VectorIndex` represents a set of `Documents` that can be interacted with using the developer-friendly `query`, `similar` and `search` APIs.

A simple implementation of `VectorIndex` implements one method; `get_documents`, returning an iterable of `Document` objects that are to be stored in the index.

The `rebuild_index` method on a `VectorIndex` takes those documents and stores them in a Vector Backend.

## `Backend` - `backends/__init__.py`

`Backend`s represent some system where `Document`s can be stored. They're an abstraction that allows us to support multiple ways of managing embeddings ranging from `NumpyBackend` where everything is managed in-memory, through `PgVectorBackend` which uses your existing PostgreSQL database, to `WeaviateBackend` which enables support for SaaS vector database products.

A `VectorIndex` uses the `Backend` configured as `default` in the Django settings by default, but it can be overriden on a per-`VectorIndex` level so that multiple `Backend`s can be used in a single project.

## `Document` - `index/base.py`

`Document`s are a dataclass represent something that is stored in a `VectorIndex`. They have an identifier, a vector (for the embedding) and an unstructured metadata dict. This class allows us to store anything in a `VectorIndex` without needing to build indexes that hold specific types of object.

Whenever we are working with `VectorIndex`s, we are working with `Document` objects but as a user implementing `wagtail-vector-index`, these `Document`s aren't usually what we want to be working with. We would prefer to deal with our models and `Page`s, and let the package transparently handle converting them back and forth to `Document`s.

This is where `DocumentConverter`s come in.

## `DocumentConverter` - `converter.py`

`DocumentConverter` is a protocol that defines how to convert an object to `Document`s and `Document`s back to an object.

To go from an object to a `Document` is usually a case of:

1. Determining a representation of the object that should be embedded
2. Splitting that representation up in to chunks to fit within the the embedding model's limit
3. Generating embeddings for each chunk
4. Returning one or more `Document` objects containing the embedding and some metadata about the original object

To go from a `Document` back to an object we have to rely on the `Document` `metadata`. This could be something like an `id` or `UUID` which will enable us to retrieve the original object from a database/filesystem, or it could be more complex metadata allowing us to reconstruct the object.

## Model-specific implementations - `index/models.py`

While all of the above are intended to be generic and usable for any object type, the main use-case for `wagtail-vector-index` is to index Django models or Wagtail Pages.

For this, we implement specialised versions of these classes/protocols and some utilities around them that are more likely to be consumed by developers.

* `Embedding` is a Django model which stores embeddings in the application database, ignoring the presence of any specific Vector Backend. This enables quickly repopulating vector backends, as well as some performance optimisations as we can get use generic foreign keys to return our related model instances.
* `EmbeddableFieldsMixin` is a way to let developers specify what fields of their model they want to index by adding the mixin and adding `embedding_fields` to a model. This doesn't do anything interesting by itself.
* `EmbeddableFieldsDocumentConverter` knows how to convert anything with the `EmbeddableFieldsMixin` to a document, and when instantiated with a `base_model`, knows how to convert `Documents` back to that `base_model`.
* `ModelVectorIndex` can be subclassed with a list of `QuerySet`s of models with `EmbeddableFieldsMixin` and manages the index for them. It uses `EmbeddableFieldsDocumentConverter` to shepherd documents back and forth.
* `GeneratedIndexMixin` is a convenience mixin which allows a developer to call `get_vector_index` on their model to return an automatically generated `ModelVectorIndex`.
* `VectorIndexedMixin` combines `GeneratedIndexMixin` and `EmbeddableFieldsMixin` to create a single mixin that developers can use to easily implement `wagtail-vector-index` features without needing to know the underlying mixins.

## In Summary

- `VectorIndex`s are responsible for fetching all the documents to be indexed, and for the interfaces for searching those documents. They have a `converter_class` class variable which specifies which `DocumentConverter` to ues for shepherding `Document`s.
- `DocumentConverter`s convert `Document`s to and from the type the user is dealing with. They might need to be specific to a certain model, or they could be written in a more generic way to convert based on metadata in the `Document`.
