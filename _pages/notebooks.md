---
layout: archive
title: "Notebooks"
permalink: /notebooks/
author_profile: false
---
Notebooks are best viewed in a desktop browser.

{% include base_path %}

{% for post in site.notebooks reversed %}
  {% include archive-single.html %}
{% endfor %}
