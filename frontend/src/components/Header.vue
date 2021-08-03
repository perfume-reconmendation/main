<template>
  <div>{{ current_page + 1 }} / 5</div>
</template>

<script>
export default {
  props: {
    current_page: {
      type: Number,
    },
  },
};
</script>
<style scoped>
div {
  font-size: 24px;
}
</style>
