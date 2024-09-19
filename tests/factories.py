import factory
import wagtail_factories
from faker import Faker
from testapp.models import BookPage, FilmPage, VideoGame
from wagtail_vector_index.storage.models import Embedding

fake = Faker()


class VideoGameFactory(factory.django.DjangoModelFactory):
    title = factory.Faker("sentence")
    description = factory.LazyFunction(lambda: "\n".join(fake.paragraphs()))

    class Meta:
        model = VideoGame


class BookPageFactory(wagtail_factories.PageFactory):
    class Meta:
        model = BookPage

    title = factory.Faker("sentence")
    body = factory.LazyFunction(lambda: "\n".join(fake.paragraphs()))


class FilmPageFactory(wagtail_factories.PageFactory):
    class Meta:
        model = FilmPage

    title = factory.Faker("sentence")
    description = factory.LazyFunction(lambda: "\n".join(fake.paragraphs()))


class EmbeddingFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Embedding

    vector = factory.LazyFunction(lambda: [fake.pyfloat() for _ in range(300)])
    content = factory.LazyFunction(lambda: "\n".join(fake.paragraphs()))
