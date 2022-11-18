{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
..
   HACK -- meth the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nohidden:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
..
   HACK -- attr the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :hidden:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
