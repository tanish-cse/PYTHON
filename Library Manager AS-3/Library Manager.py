import json
from pathlib import Path
import logging

# Setting up logging (basic)
logging.basicConfig(filename="library.log", level=logging.INFO)


# -----------------------------
# Book Class (Task 1)
# -----------------------------
class Book:
    def __init__(self, title, author, isbn, status="available"):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.status = status  # available or issued

    def __str__(self):
        return f"{self.title} by {self.author} | ISBN: {self.isbn} | Status: {self.status}"

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "isbn": self.isbn,
            "status": self.status
        }

    def issue(self):
        if self.status == "available":
            self.status = "issued"
            return True
        return False

    def return_book(self):
        self.status = "available"

    def is_available(self):
        return self.status == "available"


# -----------------------------
# Inventory Manager (Task 2 & 3)
# -----------------------------
class LibraryInventory:
    def __init__(self, file_path="book_catalog.json"):
        self.file_path = Path(file_path)
        self.books = []
        self.load_data()

    def load_data(self):
        try:
            if self.file_path.exists():
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        self.books.append(Book(**item))
            else:
                self.save_data()
        except Exception as e:
            logging.error(f"Error loading catalog: {e}")
            print("⚠️ Could not load saved data. Starting fresh.")

    def save_data(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump([b.to_dict() for b in self.books], f, indent=4)
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def add_book(self, book):
        self.books.append(book)
        self.save_data()

    def search_by_title(self, title):
        return [b for b in self.books if title.lower() in b.title.lower()]

    def search_by_isbn(self, isbn):
        return next((b for b in self.books if b.isbn == isbn), None)

    def display_all(self):
        return self.books


# -----------------------------
# CLI Menu (Task 4 & 5)
# -----------------------------
def menu():
    print("\n======== LIBRARY INVENTORY MANAGER ========")
    print("1. Add Book")
    print("2. Issue Book")
    print("3. Return Book")
    print("4. View All Books")
    print("5. Search Book")
    print("6. Exit")
    print("===========================================")


def run_library():
    inventory = LibraryInventory()

    while True:
        menu()
        choice = input("Enter your choice: ")

        try:
            if choice == "1":
                title = input("Enter title: ")
                author = input("Enter author: ")
                isbn = input("Enter ISBN: ")
                inventory.add_book(Book(title, author, isbn))
                print("✔ Book added successfully!")

            elif choice == "2":
                isbn = input("Enter ISBN to issue: ")
                book = inventory.search_by_isbn(isbn)
                if book and book.issue():
                    inventory.save_data()
                    print("✔ Book issued successfully.")
                else:
                    print("⚠️ Book not found or already issued.")

            elif choice == "3":
                isbn = input("Enter ISBN to return: ")
                book = inventory.search_by_isbn(isbn)
                if book:
                    book.return_book()
                    inventory.save_data()
                    print("✔ Book returned successfully.")
                else:
                    print("⚠️ Invalid ISBN.")

            elif choice == "4":
                books = inventory.display_all()
                if not books:
                    print("No books available.")
                else:
                    print("\n--- Library Books ---")
                    for b in books:
                        print(b)

            elif choice == "5":
                title = input("Enter title keyword: ")
                results = inventory.search_by_title(title)
                if results:
                    for r in results:
                        print(r)
                else:
                    print("⚠️ No matching book found.")

            elif choice == "6":
                print("Thank you! Exiting program.")
                break

            else:
                print("⚠️ Invalid option. Try again.")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print("Something went wrong. Please try again.")


# -----------------------------
# Program Entry Point
# -----------------------------
if __name__ == "__main__":
    run_library()