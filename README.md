# METHODS ON THE ESTIMATIONS OF SOFR TERM STRUCTURES

This is the introduction of the Master thesis in FE-900. In the thesis, I introduced several different kinds of factor models to estimate the term structure of SOFR term rate. For different models, different data sets were applied to the real operation.

**Simple Factor Model:** We obtained data from Bloomberg to determine the SOFR term rate. We had the real prices for one- and three-month contracts for the years 2020/04/01 to 2021/03/31. As with the SOFR, Bloomberg provides daily SOFR rate data. The meeting information is then obtained on the Federal Reserve's website at \url{https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm}.  This website contains information on the term's announcement date and some of the policies that were announced throughout the term period.

**Nelson-Siegel Model:** Data was received from Bloomberg and compiled the SOFR rate for various tenors available in the market. The data spans the years 2018/06/01 to 2021/07/02. Our data is collected on a weekly basis to avoid an unequal segment between each of the two closure dates.
Seven tenors were gathered from Bloomberg in total. There are six durations available: overnight, one week, one month, two months, three months, six months, and twelve months. What is worthy of our attention is the NA value contained inside the data.

