# Using Vector Indexes

Once you've got an instance of a Vector Index, either through `self.vector_index` on a `VectorIndexedMixin` model, or through your own custom index, you can run a few useful operations on the index:

### Natural language question/answers

The `query` method can be used to ask natural language questions:

```python
index.query("What is the airspeed velocity of an unladen swallow?")

QueryResponse(
    response="What do you mean? An African or a European swallow?", sources=[MyPage(1)]
)
```

Behind the scenes, this:

1. Converts the query in to an embedding
2. Uses the vector backend to find content in the same index that is similar
3. Merges all the matched content in to a single 'context' string
4. Passes the 'context' string along with the original query to the AI backend.
5. Returns a `QueryResponse` containing the `response` from the AI backend, and `sources`,
   a list of objects that were used as context.

### Getting similar content

The `similar` index method can be used to find model instances that are similar to another instance:

```python
index.similar(my_model_instance)

[MyPage(1), MyPage(2)]
```

This works by:

1. Generating (or retrieving existing) embeddings for the instance
2. Using the vector database to find matching embeddings
3. Returning the original model instances that were used to generate these matching embeddings

### Searching content

The `search` index method can be used to use natural language to search content in the index.

```python
index.search("Bring me a shrubbery")

[MyPage(1), MyPage(2)]
```

This is similar to querying content, but it only returns content matches without a natural language response.
