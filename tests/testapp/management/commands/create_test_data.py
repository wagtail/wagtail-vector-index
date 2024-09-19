import json

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from wagtail.models import Page

from testapp.models import Book, Film, MediaIndex, VideoGame


class Command(BaseCommand):
    def import_books(self, root_page: Page):
        book_index = MediaIndex(title="Books")
        root_page.add_child(instance=book_index)

        with open("tests/testapp/fixtures/books_data.json") as f:
            books = json.load(f)
            for book in books:
                book_page = Book(title=book["title"], body=book["description"])
                book_index.add_child(instance=book_page)

    def import_films(self, root_page: Page):
        film_index = MediaIndex(title="Films")
        root_page.add_child(instance=film_index)

        with open("tests/testapp/fixtures/films_data.json") as f:
            films = json.load(f)
            for film in films:
                film_page = Film(title=film["title"], description=film["description"])
                film_index.add_child(instance=film_page)

    def import_video_games(self):
        with open("tests/testapp/fixtures/games_data.json") as f:
            games = json.load(f)
            for game in games:
                VideoGame.objects.create(
                    title=game["title"], description=game["description"]
                )

    def create_superuser(self):
        User.objects.create_superuser(
            username="admin", email="admin@example.com", password="admin"
        )

    def handle(self, *args, **options):
        root_page = Page.objects.get(pk=2)
        self.import_books(root_page)
        self.import_films(root_page)
        self.import_video_games()
        self.create_superuser()
