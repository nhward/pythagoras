---
output:
  html:
    meta:
      css: ["@tabsets"]
      js: ["@tabsets"]
---
This card explores the possible expansions of temporal data. Variables
such as a date cannot be directly used in modeling. They need to be
converted into something numeric or nominal. Another way of saying this
is that we are seeking to encode the temporal data.

The types of temporal data that we are looking to encode are:

| Type            | Example                | Class          |
|:---------------:|:----------------------:|:--------------:|
| Simple Dates    | 1995-05-23             | Date           |
| Date & Times    | 1995-05-23<br>16:44:00 | POSIXt         |
| Quarters        | Q2-1995                | yearquarter    |
| Month           | 07-1995                | yearmonth      |
| Weeks           | W15-1995               | yearweek       |
| Quarters        | Q2-1995                | yearquarter    |
| Time            | 16:44:00               | hms            |

There are some issues to consider about this.

-   How does daylight-saving affect times?
-   What time-zone does the date/time apply to?
-   What format should the dates and times be displayed as? AM/PM versus
    24 hour clock? mm/dd/yyyy or dd/mm/yy or yyyy/mm/dd

# Encodings {.tabset .tabset-fade .tabset-pills}

Encodings fall into different types. The encoding to a number (integer
or double) is the first. Encoding to pairs of cyclic (double) variables
is the second. Encoding to sets of binary variables is the last.


## Numerical encodings

Since dates (and date&times) can be thought of as the numbers of days
since some arbitrary origin, it is simple to encode dates and times into
a number (whole numbers for dates, decimal values for date&times). The
arbitrary origin is fixed at 1970-01-01 00:00:00

Times can be thought of as fragments of a day especially if they are
part of a date&time variable. Times such as "hms" (which stands for
Hours Minutes Seconds) is recorded in seconds rather than days since
they are usually relative to midnight (00:00:00.00).

Thus numerical embedding, should be thought of as an interval-scaled
variable.

#### Numeric chart

The _Numeric_ tab in the card shows box-plots of any numerical encodings.
The buttons determine which types of temporal variable to encode in this
way.

## Harmonic encodings

Harmonic encoding are also known as cyclic / modulo / sinusoidal
encoding.

Dates can be interpreted as days-of-the-week. This forms an ordinal
factor (say Monday to Sunday). Dates can also be interpreted as
weeks-of-the-year, months-of-the-year, quarters-of-the-year. Times can
be interpreted as 24-hours-of-the-day, 12-hours-of-the-day,
minutes-of-the-hour, seconds-of-the-minute.

These all represent cycles in that they repeat. This makes the question
"What is the distance between Monday and Friday?" have two possible
answers. 3 days or 2 days.

Any variable that can be drawn on a circle makes a viable harmonic
variable.

Our treatment, that overcomes the "two possible answers" problem, is to
map the levels onto a circle and take their X & Y coordinates (using
_sine_s and _cosine_s of their angles).

Since a circle is round there is no obvious top. This again means that
Harmonic encoding variables (in their sine and cosine parts) are
interval scaled variables.

#### Example

Suppose days of the week were mapped to a circle. 

![Clock with days of the week](www/clock.jpeg)

With Sunday at the top, (and zero degrees at the top) the angles are:

|Day of week | Angle&deg; |  Sine  |  Cosine  |
|:------------:|:---------:|:--------:|:----------:|
| Sunday     |   0     | 0.000  | 1.000    |
| Monday     |  51     | 0.782  | 0.623    |
| Tuesday    |  103    | 0.975  |-0.223    |
| Wednesday  |  154    | 0.434  |-0.901    |
| Thursday   |  206    |-0.434  |-0.901    |
| Friday     |  257    |-0.975  |-0.223    |
| Saturday   |  309    |-0.782  | 0.623    |


The distance between Sunday and Friday calculated as the euclidean distance between the sines and cosines of the two days i.e.
$distance^2 = (Monday_{sin} - Friday_{sin})^2 + (Monday_{cos} - Friday_{cos})^2$ 

The distance is $sqrt((0.000 - -0.975)^2 + (1.000 - -0.223)^2 ) = 1.564$

This result is unambiguous and does not depend which day is assigned to zero degrees.

#### Harmonic chart

The _Harmonic_ tab in the card shows x-y scatter plots of any sinusoids. These sinusoids produce ring-like structures. 
Each encoded variable is allocated its own radius. The full-screen version of the chart provides a legend.

The buttons determine which types of temporal variable to encode in this way.

## Binary encodings

Binary encoding arise from in two ways:

 * From logical variables e.g. whether time is AM or PM
 * From nominal factors that are dummy encoded to a set of binary variables e.g. which holiday a date falls upon

#### Binary chart

The _Binary_ tab in the card shows bar plots of any binary encoded variables.

The buttons determine which types of temporal variable to encode in this way.
 
#

### Holidays

Using the settings it is possible to mark an interest in particular holidays which will be reflected in the binary chart.
 
#### Interval-scale

An interval scaled variable is a type of quantitative data that has a
clear order, with equal and meaningful distances between values, but
lacks a true zero point. This means you can meaningfully perform
addition and subtraction to find differences but not true ratios, as the
zero is arbitrary and doesn't represent the complete absence of the
variable. Common examples also include temperature in Celsius or
Fahrenheit, where zero doesn't mean "no temperature".

See [Levels of measurement](https://en.wikipedia.org/wiki/Level_of_measurement#Interval_scale)

### Scope

Note that many cyclic phenomenon have **not** been encoded. These include:

 - Season (spring, summer, autumn/fall, winter, wet, dry)
 - Tidal phase
 - Lunar phase (waxing, full, waning, none)
 - Solar cycle (re: sun spot frequency) 

The first two require a knowledge of location as well at time. 

### Tidy

The tidy button being ticked performs some clean-up operations:

 * _Remove ordinal factors:_ The process of converting the harmonic variables to sinusoidal components will remove the original ordinal factor variable.
 * _Remove sparse encodings:_ Any new variables are tested for sparsity. If during resampling, there is 
 a reasonable chance of a subset that has zero variance, the variable is dropped. 
 This is a form of variable selection based not on variable importance, but on its sampling reliability.
 
