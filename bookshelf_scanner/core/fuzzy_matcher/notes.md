- Add a convenience method that can grab results greater than or equal to a confidence (100%)
- Add a convenience method to export to different formats (like CSV)
- Add a convenience method to combine all of the bounded boxes in an image into one string (with a toggleable confidence)
- Test title and author combinations
- More efficient storage of the full array of titles

Tiered matching:
- Run the first convenience method on title alone (and take the difference with unmatched images)
- Run the first convenience method on title and author (and take the difference with unmatched images)
- Run the approval process on remaining images that meet a confidence threshold overall (say 70% or higher)
  - Show all options above that %ish
  