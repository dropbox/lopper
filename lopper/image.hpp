#pragma once

namespace lopper {

/* A lightweight image container typed by the underlying primitive.
 * Inherit from this class, and implement Expr(...) if you'd like to pass your own image containers directly to Lopper.
 * This abstraction assumes that pixels are row-major, column-minor, with interleaved channels, and that
 * the data is contiguous within each row.
 */
template<typename T> class _Image {
public:
  virtual ~_Image() {};
  /* Returns the width of the underlying image. */
  virtual int getWidth() const = 0;
  /* Returns the height of the underlying image. */
  virtual int getHeight() const = 0;
  /* Returns the # of channels in the underlying image. */
  virtual int getChannelCount() const = 0;
  /* Returns the pointer to the first column of the given row. */
  virtual T* getRowPointer(const size_t) = 0;
  virtual const T* getRowPointer(const size_t) const = 0;
};

} // end namespace lopper
