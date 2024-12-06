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

The `find_similar` index method can be used to find model instances that are similar to another instance:

```python
index.find_similar(my_model_instance)

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

## Using Filters with Vector Indexes

Wagtail Vector Index provides a filtering mechanism that allows you to refine the results of your vector searches. Filters can be applied to limit the scope of your queries, searches, and similarity lookups.

### Available Filters

There are two built-in filters available:

1. `QuerySetFilter`: Filters documents based on a given QuerySet of objects found in your index.
2. `ObjectTypeFilter`: Filters documents based on specific object types found in your index.

### Applying Filters

You can apply filters to a Vector Index using the `filter` method. This method returns a new Vector Index instance with the filters applied.

```python
from wagtail_vector_index.storage.filters import QuerySetFilter, ObjectTypeFilter
from myapp.models import MyPage, MyOtherPage

# Create a filter based on a QuerySet
queryset_filter = QuerySetFilter(MyPage.objects.filter(title__contains="AI"))

# Create a filter based on model types
object_type_filter = ObjectTypeFilter(MyPage, MyOtherPage)

# Apply filters to your index
filtered_index = index.filter(queryset_filter, object_type_filter)
```

### Using Filtered Indexes

Once you've applied filters to your index, you can use the filtered index just like a regular index. All subsequent operations on this filtered index will respect the applied filters.

```python
# Query the filtered index
response = filtered_index.query("How do you know she is a witch?")

# Search the filtered index
results = filtered_index.search("Holy Hand Grenade of Antioch")

# Find similar content within the filtered index
similar_pages = filtered_index.find_similar(my_page_instance)
```

You can also chain operations to filter and search in one line:

```python
# Chain operations to filter and search in one line
my_filter = QuerySetFilter(MyPage.objects.live())
results = index.filter(my_filter).search("How to build a Trojan Rabbit")
```

### Combining Filters

You can apply multiple filters to an index. When multiple filters are applied, they work in conjunction (AND logic), further refining the results.

```python
filtered_index = index.filter(
    QuerySetFilter(MyPage.objects.filter(published=True)),
    ObjectTypeFilter(MyPage, MyOtherPage),
)
```

In this example, the resulting index will only include published pages of type `MyPage` or `MyOtherPage`.

### Custom Filters

If you need more complex filtering logic, you can create custom filters by implementing the `DocumentFilter` protocol. Your custom filter should implement an `apply` method that takes a `DocumentQuerySet` and returns a filtered `DocumentQuerySet`.

```python
from wagtail_vector_index.storage.filters import DocumentFilter
from wagtail_vector_index.storage.models import DocumentQuerySet


class MyCustomFilter(DocumentFilter):
    def apply(self, documents: DocumentQuerySet) -> DocumentQuerySet:
        # Implement your custom filtering logic here
        return documents.filter(...)


# Use your custom filter
custom_filter = MyCustomFilter()
filtered_index = index.filter(custom_filter)
```
