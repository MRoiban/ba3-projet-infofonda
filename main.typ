#import "@preview/equate:0.2.0": equate
#import "@preview/showybox:2.0.1": showybox
#import "@preview/lovelace:0.3.0": *
#import "@preview/ilm:1.1.2": *

#show: ilm.with(
  title: [Title],
  author: "Roiban Marius-Alexandru",
  date: datetime(year: 2024, month: 07, day: 22),
  abstract: [],
  preface: [],
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
  // table-of-contents: (none),
)

#let showbox(title, body) = box()[
  #showybox(
    shadow: (
      offset:3.5pt
    ), title: title,
    [#body]
  )
]


#let pseudocodeblock(title, code) = box[#text(
  font: "FiraCode Nerd Font Mono", 
  ligatures: true
)[
  #showybox(
    shadow: (
      offset: 3.5pt
  ),
  title: title,
  [#code]
  )
]]

#let CQFD = box[$space square.filled$]
