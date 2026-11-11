package benchmark

import "fmt"

// codeFile is one source file the code workload (PromptStyleCode) asks
// the model to edit: rename Method to Rename everywhere, add type hints,
// and return the whole file.
type codeFile struct {
	// Name is the file's name, used for the extra copies that pad a
	// prompt longer than the file.
	Name   string
	Method string
	Rename string
	Body   string
}

// benchCodeFiles are the files the code workload edits, one per
// repetition. Each repetition edits a different file because n-gram
// speculative decoding (ngram-mod, ngram-cache) keeps what it has seen
// in the server between requests. Editing the same file every time, the
// second repetition drafted the answer the first had written, and
// generation sped up run over run: on compute2, 27% faster on the second
// repetition and 33% on the third, against no change for any other
// setting. The spread that made looked like noise and buried real
// differences between settings. A different file every time leaves each
// edit as new as a user's would be; every setting measures the same
// three files, so the comparison stays fair.
//
// The files are of a size and kind: plain classes, no imports, a method
// to rename that a module function at the end calls too.
var benchCodeFiles = []codeFile{
	{Name: "inventory", Method: "add_item", Rename: "add_stock", Body: benchCodeInventory},
	{Name: "library", Method: "lend_book", Rename: "check_out", Body: benchCodeLibrary},
	{Name: "timesheet", Method: "log_time", Rename: "record_time", Body: benchCodeTimesheet},
}

// BenchPromptCodePrefixTemplate asks for an edited copy of the file that
// follows: the repetition number, then the method to rename and its new
// name. The edits are mechanical and spread through the file, so the
// answer repeats most of the input while still being written rather than
// recalled.
const BenchPromptCodePrefixTemplate = "This is benchmark repetition number %d. Rename the method `%s` to `%s` everywhere it appears, add type hints to every function, and return the complete updated file with no commentary.\n\n"

// codeFileFor is the file a repetition edits. Repetitions count from 1.
// A preset with more repetitions than files starts over, and its later
// repetitions are no longer new to an n-gram cache.
func codeFileFor(repetition int) codeFile {
	i := (repetition - 1) % len(benchCodeFiles)
	if i < 0 {
		i += len(benchCodeFiles)
	}
	return benchCodeFiles[i]
}

// codePrefix is the instruction for one repetition's edit.
func codePrefix(repetition int) string {
	f := codeFileFor(repetition)
	return fmt.Sprintf(BenchPromptCodePrefixTemplate, repetition, f.Method, f.Rename)
}

// codeText is a repetition's file as the prompt carries it.
func codeText(repetition int) string {
	return "```python\n" + codeFileFor(repetition).Body + "```\n"
}

const benchCodeInventory = `"""Inventory tracking for a small warehouse.

The module keeps stock levels per item, records movements in and out,
and reports on what needs reordering. It is deliberately plain: a
dictionary of items, a list of movements, and functions over them.
"""


class Item:
    """A single stocked product."""

    def __init__(self, sku, name, unit_price, reorder_level=0):
        self.sku = sku
        self.name = name
        self.unit_price = unit_price
        self.reorder_level = reorder_level
        self.quantity = 0

    def value(self):
        """Return the value of the stock held for this item."""
        return self.quantity * self.unit_price

    def needs_reorder(self):
        """Return True when stock has fallen to the reorder level."""
        return self.quantity <= self.reorder_level

    def __repr__(self):
        return "Item(sku=%r, name=%r, quantity=%r)" % (self.sku, self.name, self.quantity)


class Movement:
    """A change in stock: positive in, negative out."""

    def __init__(self, sku, quantity, reason, at):
        self.sku = sku
        self.quantity = quantity
        self.reason = reason
        self.at = at

    def is_inbound(self):
        """Return True when this movement added stock."""
        return self.quantity > 0


class Inventory:
    """The warehouse's stock, indexed by SKU."""

    def __init__(self, name):
        self.name = name
        self.items = {}
        self.movements = []

    def register(self, item):
        """Add an item to the catalogue, replacing one with the same SKU."""
        self.items[item.sku] = item
        return item

    def add_item(self, sku, quantity, reason="delivery", at=None):
        """Add stock for one SKU and record the movement.

        Raises KeyError when the SKU is not in the catalogue, and
        ValueError when the quantity is not positive.
        """
        if sku not in self.items:
            raise KeyError("unknown sku: %s" % sku)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        item = self.items[sku]
        item.quantity += quantity
        self.movements.append(Movement(sku, quantity, reason, at))
        return item.quantity

    def remove_item(self, sku, quantity, reason="sale", at=None):
        """Take stock out for one SKU and record the movement."""
        if sku not in self.items:
            raise KeyError("unknown sku: %s" % sku)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        item = self.items[sku]
        if item.quantity < quantity:
            raise ValueError("only %d of %s in stock" % (item.quantity, sku))
        item.quantity -= quantity
        self.movements.append(Movement(sku, -quantity, reason, at))
        return item.quantity

    def quantity_of(self, sku):
        """Return the stock held for one SKU, or zero when unknown."""
        item = self.items.get(sku)
        return item.quantity if item else 0

    def total_value(self):
        """Return the value of everything in stock."""
        return sum(item.value() for item in self.items.values())

    def reorder_list(self):
        """Return the items at or below their reorder level, by name."""
        low = [item for item in self.items.values() if item.needs_reorder()]
        return sorted(low, key=lambda item: item.name)

    def movements_for(self, sku):
        """Return every movement recorded for one SKU, oldest first."""
        return [m for m in self.movements if m.sku == sku]

    def busiest_items(self, limit=5):
        """Return the SKUs with the most movements, busiest first."""
        counts = {}
        for movement in self.movements:
            counts[movement.sku] = counts.get(movement.sku, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]

    def summary(self):
        """Return a short report on the state of the inventory."""
        lines = ["Inventory: %s" % self.name]
        lines.append("  items: %d" % len(self.items))
        lines.append("  total value: %.2f" % self.total_value())
        low = self.reorder_list()
        if low:
            lines.append("  needs reorder: %s" % ", ".join(item.name for item in low))
        return "\n".join(lines)


def restock_from_plan(inventory, plan, reason="plan"):
    """Apply a {sku: quantity} plan to an inventory, skipping unknown SKUs."""
    applied = {}
    for sku, quantity in plan.items():
        if sku not in inventory.items:
            continue
        applied[sku] = inventory.add_item(sku, quantity, reason=reason)
    return applied
`

const benchCodeLibrary = `"""Loans for a small lending library.

The module keeps the catalogue of books, the members who may borrow
them, and the loans between the two. It is deliberately plain: a
dictionary of books, a dictionary of members, and functions over them.
"""


class Book:
    """A single copy of a title the library can lend."""

    def __init__(self, isbn, title, author, loan_days=21):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.loan_days = loan_days
        self.on_loan = False

    def is_available(self):
        """Return True when the book is on the shelf."""
        return not self.on_loan

    def label(self):
        """Return the title and author as the shelf label shows them."""
        return "%s, by %s" % (self.title, self.author)

    def __repr__(self):
        return "Book(isbn=%r, title=%r, on_loan=%r)" % (self.isbn, self.title, self.on_loan)


class Loan:
    """One book lent to one member, from a day to a due day."""

    def __init__(self, isbn, member_id, start_day, due_day):
        self.isbn = isbn
        self.member_id = member_id
        self.start_day = start_day
        self.due_day = due_day
        self.returned_day = None

    def is_open(self):
        """Return True while the book has not come back."""
        return self.returned_day is None

    def days_late(self, today):
        """Return how many days past the due day the loan has run."""
        end = self.returned_day if self.returned_day is not None else today
        return max(0, end - self.due_day)


class Library:
    """The library's books and members, indexed by ISBN and member ID."""

    def __init__(self, name, max_loans=5):
        self.name = name
        self.max_loans = max_loans
        self.books = {}
        self.members = {}
        self.loans = []

    def add_book(self, book):
        """Add a book to the catalogue, replacing one with the same ISBN."""
        self.books[book.isbn] = book
        return book

    def join(self, member_id, name):
        """Register a member, keeping the name already on file if any."""
        self.members.setdefault(member_id, name)
        return member_id

    def lend_book(self, isbn, member_id, today):
        """Lend one book to one member and record the loan.

        Raises KeyError when the book or the member is unknown, and
        ValueError when the book is out or the member has too many.
        """
        if isbn not in self.books:
            raise KeyError("unknown isbn: %s" % isbn)
        if member_id not in self.members:
            raise KeyError("unknown member: %s" % member_id)
        book = self.books[isbn]
        if book.on_loan:
            raise ValueError("%s is already on loan" % book.title)
        if len(self.open_loans_for(member_id)) >= self.max_loans:
            raise ValueError("member %s has %d books" % (member_id, self.max_loans))
        book.on_loan = True
        loan = Loan(isbn, member_id, today, today + book.loan_days)
        self.loans.append(loan)
        return loan

    def give_back(self, isbn, today):
        """Close the open loan for one book and put it back on the shelf."""
        for loan in self.loans:
            if loan.isbn == isbn and loan.is_open():
                loan.returned_day = today
                self.books[isbn].on_loan = False
                return loan
        raise ValueError("%s is not on loan" % isbn)

    def open_loans_for(self, member_id):
        """Return the loans a member has not yet returned."""
        return [l for l in self.loans if l.member_id == member_id and l.is_open()]

    def overdue(self, today):
        """Return the open loans past their due day, most overdue first."""
        late = [l for l in self.loans if l.is_open() and l.days_late(today) > 0]
        return sorted(late, key=lambda loan: loan.days_late(today), reverse=True)

    def popular_titles(self, limit=5):
        """Return the titles lent most often, most popular first."""
        counts = {}
        for loan in self.loans:
            title = self.books[loan.isbn].title
            counts[title] = counts.get(title, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]

    def summary(self, today):
        """Return a short report on the state of the library."""
        lines = ["Library: %s" % self.name]
        lines.append("  books: %d" % len(self.books))
        lines.append("  members: %d" % len(self.members))
        lines.append("  on loan: %d" % sum(1 for b in self.books.values() if b.on_loan))
        late = self.overdue(today)
        if late:
            lines.append("  overdue: %s" % ", ".join(self.books[l.isbn].title for l in late))
        return "\n".join(lines)


def lend_reading_list(library, member_id, isbns, today):
    """Lend every available book on a list to one member, skipping the rest."""
    lent = []
    for isbn in isbns:
        book = library.books.get(isbn)
        if book is None or not book.is_available():
            continue
        lent.append(library.lend_book(isbn, member_id, today))
    return lent
`

const benchCodeTimesheet = `"""Time tracking for a small consulting team.

The module keeps the projects the team bills, the hours each person
logs against them, and the reports drawn from those hours. It is
deliberately plain: a dictionary of projects, a list of entries, and
functions over them.
"""


class Project:
    """A piece of client work that hours are logged against."""

    def __init__(self, code, client, hourly_rate, budget_hours=0):
        self.code = code
        self.client = client
        self.hourly_rate = hourly_rate
        self.budget_hours = budget_hours
        self.closed = False

    def cost_of(self, hours):
        """Return what a number of hours on this project bills."""
        return hours * self.hourly_rate

    def has_budget(self):
        """Return True when the project has a limit on its hours."""
        return self.budget_hours > 0

    def __repr__(self):
        return "Project(code=%r, client=%r, closed=%r)" % (self.code, self.client, self.closed)


class Entry:
    """Hours one person worked on one project on one day."""

    def __init__(self, code, person, day, hours, note):
        self.code = code
        self.person = person
        self.day = day
        self.hours = hours
        self.note = note

    def is_overtime(self, limit=8):
        """Return True when the entry runs past a normal working day."""
        return self.hours > limit


class Timesheet:
    """The team's projects and logged hours, indexed by project code."""

    def __init__(self, team):
        self.team = team
        self.projects = {}
        self.entries = []

    def open_project(self, project):
        """Add a project, replacing one with the same code."""
        self.projects[project.code] = project
        return project

    def log_time(self, code, person, day, hours, note=""):
        """Record hours against one project and return its new total.

        Raises KeyError when the project is unknown, and ValueError when
        the project is closed or the hours are not between 0 and 24.
        """
        if code not in self.projects:
            raise KeyError("unknown project: %s" % code)
        project = self.projects[code]
        if project.closed:
            raise ValueError("project %s is closed" % code)
        if hours <= 0 or hours > 24:
            raise ValueError("hours must be between 0 and 24")
        self.entries.append(Entry(code, person, day, hours, note))
        return self.hours_on(code)

    def close_project(self, code):
        """Stop a project taking more hours and return what it billed."""
        if code not in self.projects:
            raise KeyError("unknown project: %s" % code)
        project = self.projects[code]
        project.closed = True
        return project.cost_of(self.hours_on(code))

    def hours_on(self, code):
        """Return the hours logged against one project, or zero."""
        return sum(e.hours for e in self.entries if e.code == code)

    def hours_by(self, person):
        """Return the hours one person has logged across every project."""
        return sum(e.hours for e in self.entries if e.person == person)

    def over_budget(self):
        """Return the projects past their budget, by project code."""
        over = [p for p in self.projects.values()
                if p.has_budget() and self.hours_on(p.code) > p.budget_hours]
        return sorted(over, key=lambda project: project.code)

    def busiest_people(self, limit=5):
        """Return the people with the most hours, busiest first."""
        totals = {}
        for entry in self.entries:
            totals[entry.person] = totals.get(entry.person, 0) + entry.hours
        ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]

    def summary(self):
        """Return a short report on the state of the timesheet."""
        lines = ["Timesheet: %s" % self.team]
        lines.append("  projects: %d" % len(self.projects))
        lines.append("  hours: %.1f" % sum(e.hours for e in self.entries))
        over = self.over_budget()
        if over:
            lines.append("  over budget: %s" % ", ".join(p.code for p in over))
        return "\n".join(lines)


def log_week(timesheet, code, person, first_day, hours_per_day, note="week"):
    """Log the same hours on five days in a row, skipping zero days."""
    totals = []
    for offset, hours in enumerate(hours_per_day[:5]):
        if hours <= 0:
            continue
        totals.append(timesheet.log_time(code, person, first_day + offset, hours, note))
    return totals
`
