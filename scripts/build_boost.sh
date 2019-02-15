wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz

tar xvf boost_1*

cd boost_*

./bootstrap.sh --with-libraries=fiber,python,program_options,test,atomic,date_time,stacktrace,thread,system,filesystem,regex,

./b2 cxxflags="-std=c++11" \
  threading=multi \
  variant=release \
  link=shared \
  define=BOOST_NO_CXX14_CONSTEXPR \
  --reconfigure

